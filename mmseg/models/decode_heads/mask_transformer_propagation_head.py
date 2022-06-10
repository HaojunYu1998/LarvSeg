from genericpath import exists
import random
import math
import os
import json
from importlib_metadata import requires
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.nn.init import trunc_normal_

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..losses.accuracy import accuracy

from mmseg.ops import resize
from mmcv.runner import force_fp32

from timm.models.layers import DropPath
from mmcv.runner import get_dist_info
from random import sample


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


@HEADS.register_module()
class MaskTransformerPropagationHead(BaseDecodeHead):

    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
        cls_emb_from_backbone=False,
        cls_emb_path="",
        cls_emb_path_test="",
        cls_emb_concat=False,
        imagenet_class_path="notebook/in21k_inter_ade_filter.json",
        imagenet_prior_loss_weight=1.0,
        imagenet_pseudo_label=False,
        imagenet_cam_thresh=0,
        pseudo_label_thresh=0.0,
        propagation_loss_weight=0.0,
        structure_loss_weight=0.0,
        downsample_rate=8,
        prior_rate=0.1,
        imagenet_prior_rate=0.1,
        grounding_inference=False,
        imagenet_pred_save_dir=None,
        temperature=1.0,
        ann_suffix=".png",
        test_anno_dir="",
        use_pairwise_affinity=False,
        pairwise_affinity_thresh=0.95,
        cam_thresh=0.9,
        call_init_weight=True,
        **kwargs,
    ):
        # in_channels & channels are dummy arguments to satisfy signature of
        # parent's __init__
        super().__init__(
            in_channels=d_encoder,
            channels=1,
            num_classes=n_cls,
            **kwargs,
        )
        del self.conv_seg
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model**-0.5
        self.propagation_loss_weight = propagation_loss_weight
        self.structure_loss_weight = structure_loss_weight
        self.prior_loss_weight = 1.0
        self.downsample_rate = downsample_rate
        self.prior_rate = prior_rate
        self.imagenet_prior_rate = imagenet_prior_rate
        self.imagenet_pseudo_label = imagenet_pseudo_label
        self.imagenet_cam_thresh = imagenet_cam_thresh
        self.pseudo_label_thresh = pseudo_label_thresh
        self.cls_emb_path = cls_emb_path
        self.cls_emb_path_test = cls_emb_path_test if len(
            cls_emb_path_test) else self.cls_emb_path
        self.cls_emb_concat = cls_emb_concat
        self.grounding_inference = grounding_inference
        self.imagenet_pred_save_dir = imagenet_pred_save_dir
        self.temperature = temperature
        self.ann_suffix = ann_suffix
        self.test_anno_dir = test_anno_dir
        # Pairwise Affinity for ImageNet21K supervision
        self.use_pairwise_affinity = use_pairwise_affinity
        self.pairwise_affinity_thresh = pairwise_affinity_thresh
        self.cam_thresh = cam_thresh

        self.call_init_weight = call_init_weight

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, dropout, dpr[i])
            for i in range(n_layers)
        ])

        self.cls_emb_from_backbone = cls_emb_from_backbone
        self.imagenet_in_batch = False
        if isinstance(cls_emb_path, list):
            self.imagenet_in_batch = any("in21k" in p for p in cls_emb_path)
            from mmcv.runner import get_dist_info
            rank, _ = get_dist_info()
            self.cls_emb_path = cls_emb_path.pop(rank % len(cls_emb_path)) #cls_emb_path[]
            self.cls_emb_other = cls_emb_path
            print(f"cuda:{rank} loading {self.cls_emb_path}.")

        if isinstance(self.cls_emb_path_test, list):
            self.cls_emb_path_test = self.cls_emb_path_test[0]
        self.loaded_cls_emb_test = False
        self.loaded_cls_emb_train = False

        self.imagenet_on_gpu = "in21k" in self.cls_emb_path if self.training \
                          else "in21k" in self.cls_emb_path_test
        # assert self.imagenet_on_gpu is True, f"{self.cls_emb_path_test}, {self.training}"
        # print(rank, self.imagenet)
        if self.imagenet_on_gpu:
            self.prior_loss_weight = imagenet_prior_loss_weight
            with open(imagenet_class_path, "r") as f:
                in21k_id_name_dict = json.load(f)
                # in21k_names = list(in21k_id_name_dict.values())
                self.in21k_ids = list(in21k_id_name_dict.keys())

        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale *
                                       torch.randn(d_model, d_model))
        # self.proj_classes = nn.Parameter(self.scale *
        #                                  torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.gamma = nn.Parameter(torch.ones([]))
        self.beta = nn.Parameter(torch.zeros([]))
        # self.mask_norm = nn.LayerNorm(n_cls)

    def init_weights(self):
        # if self.call_init_weight:
        self.apply(init_weights)
        # if not self.cls_emb_from_backbone:
        #     trunc_normal_(self.cls_emb, std=0.02)

    def forward(self, x, img_metas, img_labels=None):
        if self.cls_emb_from_backbone:
            x, cls_emb = x
        else:
            cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = self._transform_inputs(x)
        cls_emb = cls_emb.to(x.device).to(x.dtype)

        B, C, H, W = x.size()
        
        x = x.view(B, C, -1).permute(0, 2, 1)

        x = self.proj_dec(x)
        feats = x.clone()
        feats = feats.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            if img_labels is not None:
                if isinstance(img_labels[0], int):
                    img_labels = [[x] for x in img_labels]
                assert isinstance(img_labels[0], list)
                grounding_cls_emb = [
                    x[idx] for x, idx in zip(cls_emb, img_labels)
                ]
                print(grounding_cls_emb)
                x_list = []
                for b in range(len(x)):
                    x_list.append(
                        blk(x[[b]], grounding_cls_emb[[b]])
                    )
                x = torch.cat(x_list)
            else:
                x = blk(x, cls_emb)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x, cls_emb
        patches = patches @ self.proj_patch
        # cls_seg_feat = cls_seg_feat @ self.proj_classes

        # B, HW, C
        patches = patches / patches.norm(dim=-1, keepdim=True)
        # B, N, C
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        embeds = patches.clone()
        if self.training:
            # masks = self.mask_norm(masks)
            masks = (
                (masks - torch.mean(masks, dim=-1, keepdim=True))
                / torch.sqrt(torch.var(masks, dim=-1, keepdim=True, unbiased=False) + 1e-5)
            ) * self.gamma + self.beta
        B, HW, N = masks.size()

        masks = masks.view(B, H, W, N).permute(0, 3, 1, 2)
        embeds = embeds.view(B, H, W, C).permute(0, 3, 1, 2)
        return masks, embeds, feats

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        # load cls_emb, and avoid unnecessary loading
        if (not self.cls_emb_from_backbone) and (not self.loaded_cls_emb_train):
            self.cls_emb = torch.load(self.cls_emb_path, map_location="cpu")            
            self.cls_emb.requires_grad = False
            self.loaded_cls_emb_test = False
            self.loaded_cls_emb_train = True

        img_labels = None
        if self.imagenet_on_gpu:
            img_names = [meta['ori_filename'] for meta in img_metas]
            img_ids = [name[:name.find("_")] for name in img_names]
            img_labels = [self.in21k_ids.index(img_id) for img_id in img_ids]
        else:
            img_labels = [gt.unique() for gt in gt_semantic_seg]
            img_labels = [l[l != self.ignore_index].tolist() for l in img_labels]

        masks, embeds, feats = self.forward(inputs, img_metas, img_labels)
        losses = self.losses(masks, embeds, gt_semantic_seg, img_labels)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        if not self.loaded_cls_emb_test:
            self.cls_emb = torch.load(
                self.cls_emb_path_test, map_location="cpu")
            self.loaded_cls_emb_test = True
            self.loaded_cls_emb_train = False
        
        masks, _, feats = self.forward(inputs, img_metas)
        # print(feats.shape)
        # save_path = os.path.join(
        #     "/mnt/haojun/itpsea4data/OpenVocSeg/outputs/final_paper_SITSeg_ADE20K_pixel_embeddings",
        #     img_metas[0]['ori_filename'].replace("jpg", "pth")
        # )
        # torch.save(feats, save_path)
        # print(save_path)
        # img_labels = None
        # if self.imagenet_on_gpu:
        #     img_names = [meta['ori_filename'] for meta in img_metas]
        #     img_ids = [name[:name.find("_")] for name in img_names]
        #     img_labels = [self.in21k_ids.index(img_id) for img_id in img_ids]
        #     img_shapes = [meta['ori_shape'][:-1] for meta in img_metas]
        
        # os.makedirs(self.imagenet_pred_save_dir, exist_ok=True)
        # if self.imagenet_pred_save_dir is not None:
        #     pred = masks.softmax(dim=1)
        #     pred = pred[0, img_labels[0]]
        #     h, w = img_shapes[0]
        #     pred = F.interpolate(
        #         pred[None, None], size=(h, w), mode="bilinear", align_corners=False
        #     )[0,0]
        #     pred = (pred - pred.min()) / (pred.max() - pred.min())
        #     pred = pred.cpu().numpy()
        #     Image.fromarray((pred * 255).astype(np.uint8)).save(
        #         os.path.join(self.imagenet_pred_save_dir, img_names[0].replace(".jpg", ".png"))
        #     )
        
        if self.grounding_inference:
            
            if len(self.test_anno_dir) > 0:
                gt_path = os.path.join(
                    self.test_anno_dir,
                    img_metas[0]["filename"].split("/")[-1].replace(".jpg", self.ann_suffix)
                )
            else:
                gt_path = img_metas[0]["filename"].replace(
                    "images", "annotations").replace(".jpg", self.ann_suffix)
            gt = np.array(Image.open(gt_path))
            if self.ann_suffix == ".png":
                gt = (gt - 1).astype(np.uint8)
                ignore_index = 255
            else:
                gt = gt.astype(np.uint16)
                ignore_index = 65535
            unique_label = list(np.unique(gt))
            unique_label = [l for l in unique_label if l != ignore_index]
            B, N, H, W = masks.shape
            assert B == 1, f"batch {B} != 1 for inference"
            grounding_mask = torch.zeros(N).bool().to(masks.device)
            for l in unique_label:
                grounding_mask[l] = True
            masks = masks.squeeze(0)
            masks[~grounding_mask] = -100.0
            masks = masks.unsqueeze(0)
        return masks

    @force_fp32(apply_to=('seg_mask', ))
    def losses(self, seg_mask, seg_embed, seg_label, img_labels=None):
        """Compute segmentation loss."""
        loss = dict()
        h = seg_label.shape[-2] // self.downsample_rate
        w = seg_label.shape[-1] // self.downsample_rate
        seg_mask = resize(
            input=seg_mask,
            size=(h, w),
            mode='bilinear',
            align_corners=self.align_corners
        )
        seg_embed = resize(
            input=seg_embed,
            size=(h, w),
            mode='bilinear',
            align_corners=self.align_corners
        )
        seg_label = resize(
            input=seg_label.float(),
            size=(h, w),
            mode='nearest'
        ).long()

        B, N, H, W = seg_mask.shape
        if self.imagenet_on_gpu:
            prior_bucket, seg_label = self._sample_imagenet(
                seg_mask, seg_embed, seg_label, img_labels
            )
        else:
            prior_bucket = self._sample(seg_mask, seg_label)

        seg_label = seg_label.reshape(B * H * W)
        seg_mask = seg_mask.permute(0, 2, 3, 1).reshape(B * H * W, N)

        # prior_inds = None if len(prior_bucket) == 0 else torch.cat(prior_bucket)
        if len(prior_bucket) == 0:
            loss['loss_prior'] = torch.tensor(
                0, dtype=seg_mask.dtype, device=seg_mask.device, requires_grad=True
            )
            # from mmcv.runner import get_dist_info
            # rank, _ = get_dist_info()
            # print(rank, 1)
        else:
            prior_inds = torch.cat(prior_bucket)
            assert prior_inds.numel() > 0
            loss['loss_prior'] = self.loss_decode(
                seg_mask[prior_inds],
                seg_label[prior_inds],
                weight=None,
                ignore_index=self.ignore_index
            ) * self.prior_loss_weight
            # from mmcv.runner import get_dist_info
            # rank, _ = get_dist_info()
            # print(rank, 2)

        acc_weight = 0.0 if self.imagenet_on_gpu else 2.0
        acc_weight = acc_weight if self.imagenet_in_batch else 1.0
        loss['acc_seg'] = accuracy(seg_mask, seg_label) * acc_weight
        return loss

    def _sample(self, seg_mask, seg_label, min_kept=10):
        """Sample pixels that have high loss or with low prediction confidence.

        Args:
            seg_mask (torch.Tensor): segmentation logits, shape (B, N, H, W)
            seg_label (torch.Tensor): segmentation label, shape (B, 1, H, W)

        Returns:
            pos_bucket
            prior_bucket
        """
        B, N, H, W = seg_mask.size()
        seg_label = seg_label.reshape(B * H * W)
        unique_label = torch.unique(seg_label)
        unique_label = unique_label[unique_label != self.ignore_index]
        pos_bucket = []
        for l in unique_label:
            pos_bucket.append(
                (seg_label == l).nonzero(as_tuple=False)[:, 0]
            )
        if len(pos_bucket) == 0:
            return []
        seg_mask = seg_mask.permute(0, 2, 3, 1).reshape(B * H * W, N)
        num_per_bucket = []
        for p in pos_bucket:
            k = int(self.prior_rate * len(p))
            if k < min_kept:
                k = min(min_kept, len(p))
            num_per_bucket.append(k)
        prior_bucket = []
        for k, p, l in zip(num_per_bucket, pos_bucket, unique_label):
            inds = seg_mask[p, int(l)].topk(k).indices
            prior_bucket.append(p[inds])
        # print("coco", [x.shape for x in prior_bucket])
        return prior_bucket

    def _sample_imagenet(self, seg_mask, seg_embed, cam_label, img_labels):
        B, N, H, W = seg_mask.size()
        B, C, H, W = seg_embed.size()
        seg_mask = seg_mask.permute(0, 2, 3, 1).reshape(B, H * W, N).softmax(dim=-1)
        seg_embed = seg_embed.permute(0, 2, 3, 1).reshape(B, H * W, C)
        cam_label = cam_label.reshape(B, H * W)
        if self.imagenet_pseudo_label:
            prior_bucket = []
            #     cam.topk(int((cam > 0).sum())).indices + b * H * W for b, cam in enumerate(cam_label)
            # ]
            seg_label = []
            for b, cam in enumerate(cam_label):
                inds = (cam > 0).nonzero().flatten()
                cam_score = float(seg_mask[b, inds, img_labels[b]].mean())
                prior_bucket.append(inds + b * H * W)
                if cam_score > self.pseudo_label_thresh:
                    seg_label.append(torch.ones_like(cam) * img_labels[b])
                else:
                    seg_label.append(torch.ones_like(cam) * self.ignore_index)
            # print("in21k", [x.shape for x in prior_bucket])
            seg_label = torch.cat(seg_label) # B * H * W
            return prior_bucket, seg_label
        # CAM
        K = int(self.imagenet_prior_rate * H * W)
        prior_bucket = []
        for b, cam in enumerate(cam_label):
            # use threshold if thresh is defined, else use prior rate
            if self.imagenet_cam_thresh > 0:
                inds = (cam >= self.imagenet_cam_thresh).nonzero().flatten()
                # print(int(inds.numel()))
            else:
                inds = cam.topk(K).indices
            # print(float(seg_mask[b, inds, img_labels[b]].mean()))
            cam_score = float(seg_mask[b, inds, img_labels[b]].mean())
            if self.use_pairwise_affinity and (cam_score > self.cam_thresh):
                cos_sim = seg_embed[b] @ seg_embed[b, inds].T
                cos_sim = cos_sim.mean(dim=-1) # (H*W,)
                pa_inds = (cos_sim > self.pairwise_affinity_thresh).nonzero().flatten()
                inds = torch.unique(torch.cat([inds, pa_inds]))
                print(f"{cam_score:.4f}, {K}, {len(inds)}, {float(cos_sim[inds].mean()):.4f}")

            prior_bucket.append(inds + b * H * W)
                
        assert len(cam_label) == len(img_labels)
        seg_label = torch.cat([
            torch.ones_like(cam) * l for cam, l in zip(cam_label, img_labels)
        ]) # B * H * W
        return prior_bucket, seg_label

    @staticmethod
    def _get_batch_hist_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch, H, W = target.shape
        tvect = target.new_zeros((batch, nclass), dtype=torch.int64)
        for i in range(batch):
            hist = torch.histc(
                target[i].data.float(), bins=nclass, min=0, max=nclass - 1)
            tvect[i] = hist
        return tvect


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim**-0.5
        self.attn = None

        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, q, k, v, mask=None):
        B, _, C = q.shape
        # B, head, N, C // head
        q = self.q_linear(q).reshape(B, -1, self.heads,
                                     C // self.heads).permute(0, 2, 1, 3)
        k = self.k_linear(k).reshape(B, -1, self.heads,
                                     C // self.heads).permute(0, 2, 1, 3)
        v = self.v_linear(v).reshape(B, -1, self.heads,
                                     C // self.heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):

    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, cls_emb, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), cls_emb, cls_emb, mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DecoderLinear(nn.Module):

    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls

        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.head(x)
        B, HW, C = x.size()
        x = x.view(B, GS, HW // GS, C).permute(0, 3, 1, 2)
        # x = rearrange(x, "b (h w) c -> b c h w", h=GS)

        return x


class LayerScale(nn.Module):
    """Reproduced LayerNorm
    https://github.com/pytorch/pytorch/blob/e2eb97dd7682d2810071ce78b76543acc1584a9c/torch/onnx/symbolic_opset9.py#L1311
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        return self.gamma * x + self.beta
