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
class MaskTransformerLSegHead(BaseDecodeHead):

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
        imagenet_class_path=None,
        imagenet_prior_loss_weight=1.0,
        imagenet_cam_thresh=0,
        imagenet_pseudo_label=False,
        imagenet_sample_class_num=0,
        pseudo_label_thresh=0.0,
        propagation_loss_weight=1.0,
        structure_loss_weight=0.0,
        downsample_rate=8,
        prior_rate=0.1,
        imagenet_prior_rate=0.1,
        grounding_inference=False,
        imagenet_pred_save_dir=None,
        temperature=1.0,
        ann_suffix=".png",
        test_anno_dir="",
        use_pixel_embedding=False,
        use_pairwise_affinity=False,
        pairwise_affinity_thresh=0.95,
        cam_thresh=0.9,
        use_self_attention=False,
        reduce_zero_label=True,
        oracle_inference=False,
        num_oracle_points=10,
        oracle_downsample_rate=8,
        # LSeg Parameters
        block_depth=0,
        head_block_type="",
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
        self.imagenet_sample_class_num = imagenet_sample_class_num
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
        self.use_pixel_embedding = use_pixel_embedding
        # Pairwise Affinity for ImageNet21K supervision
        self.use_pairwise_affinity = use_pairwise_affinity
        self.pairwise_affinity_thresh = pairwise_affinity_thresh
        self.cam_thresh = cam_thresh
        self.use_self_attention = use_self_attention
        self.reduce_zero_label = reduce_zero_label
        self.oracle_inference = oracle_inference
        self.num_oracle_points = num_oracle_points
        self.oracle_downsample_rate = oracle_downsample_rate

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
        self.imagenet_on_gpu_ = "in11k" in self.cls_emb_path if self.training \
                          else "in11k" in self.cls_emb_path_test
        self.imagenet_on_gpu = self.imagenet_on_gpu_ or self.imagenet_on_gpu
        if self.imagenet_on_gpu:
            self.prior_loss_weight = imagenet_prior_loss_weight
            if imagenet_class_path is not None:
                with open(imagenet_class_path, "r") as f:
                    in21k_id_name_dict = json.load(f)
                    self.in21k_ids = list(in21k_id_name_dict.keys())
        self.label_cos_sim = None

        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale *
                                       torch.randn(d_model, d_model))

        self.gamma = nn.Parameter(torch.ones([]))
        self.beta = nn.Parameter(torch.zeros([]))

        # NOTE: LSeg prameters
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        self.scratch = _make_scratch(
            [768, 768, 768, 768], 768, groups=1, expand=False
        )
        self.scratch.head1 = nn.Conv2d(768, 768, kernel_size=1)

        self.block_depth = block_depth
        self.head_block_type = head_block_type
        if self.head_block_type == "bottleneck":
            self.scratch.head_block = bottleneck_block(activation=kwargs["activation"])
        elif self.head_block_type == "depthwise":
            self.scratch.head_block = depthwise_block(activation=kwargs["activation"])

        self.scratch.output_conv = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )


    def init_weights(self):
        self.apply(init_weights)

    def forward(self, x, img_metas, img_labels=None):
        assert not self.cls_emb_from_backbone
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        cls_emb = cls_emb.to(x.device).to(x.dtype)

        x = self._transform_inputs(x)
        assert len(x) == 4
        layer_1, layer_2, layer_3, layer_4 = x

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        x = self.scratch.head1(path_1)
        B, C, H, W = x.size()
        feats = x.clone()
        x = x.view(B, C, -1).permute(0, 2, 1)

        patches, cls_seg_feat = x, cls_emb
        patches = patches @ self.proj_patch

        # B, HW, C
        patches = patches / patches.norm(dim=-1, keepdim=True)
        # B, N, C
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = self.logit_scale * patches @ cls_seg_feat.transpose(1, 2)
        if self.head_block_type in ["bottleneck", "depthwise"]:
            for _ in range(self.block_depth - 1):
                masks = self.scratch.head_block(masks)
            masks = self.scratch.head_block(masks, False)
        # if self.training:
        #     masks = (
        #         (masks - torch.mean(masks, dim=-1, keepdim=True))
        #         / torch.sqrt(torch.var(masks, dim=-1, keepdim=True, unbiased=False) + 1e-5)
        #     ) * self.gamma + self.beta
        
        B, HW, N = masks.size()

        masks = masks.view(B, H, W, N).permute(0, 3, 1, 2)
        embeds = patches.clone()
        embeds = embeds.view(B, H, W, -1).permute(0, 3, 1, 2)
        return masks, embeds, feats

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        # load cls_emb, and avoid unnecessary loading
        if (not self.cls_emb_from_backbone) and (not self.loaded_cls_emb_train):
            cls_emb = torch.load(self.cls_emb_path, map_location="cpu")
            if isinstance(cls_emb, dict):
                self.in21k_ids_all = list(cls_emb.keys())
                cls_emb = torch.stack(list(cls_emb.values()))
                self.cls_emb_all = cls_emb.clone()
            self.cls_emb = cls_emb
            self.cls_emb.requires_grad = False
            self.loaded_cls_emb_test = False
            self.loaded_cls_emb_train = True

        img_labels = None
        if self.imagenet_on_gpu:
            img_names = [meta['ori_filename'].split("/")[-1] for meta in img_metas]
            img_ids = [name[:name.find("_")] for name in img_names]
            # Sample some imagenet categories
            if self.imagenet_sample_class_num > 0:
                sampled_inds = np.random.choice(
                    list(range(len(self.in21k_ids_all))), 
                    size=self.imagenet_sample_class_num, 
                    replace=False
                )
                self.in21k_ids = list(
                    set(img_ids) | set(self.in21k_ids_all[i] for i in sampled_inds)
                )
                self.cls_emb = self.cls_emb_all[
                    [self.in21k_ids_all.index(img_id) for img_id in self.in21k_ids]
                ]
            img_labels = [self.in21k_ids.index(img_id) for img_id in img_ids]
        else:
            img_labels = [gt.unique() for gt in gt_semantic_seg]
            img_labels = [l[l != self.ignore_index].tolist() for l in img_labels]
        masks, embeds, feats = self.forward(inputs, img_metas)

        losses = self.losses(masks, embeds, feats, gt_semantic_seg, img_labels)
        return losses

    def forward_test(self, inputs, img_metas, gt_semantic_seg, test_cfg):
        if not self.loaded_cls_emb_test:
            self.cls_emb = torch.load(
                self.cls_emb_path_test, map_location="cpu")
            self.loaded_cls_emb_test = True
            self.loaded_cls_emb_train = False
        
        masks, embeds, feats = self.forward(inputs, img_metas)
        if self.grounding_inference:
            if len(self.test_anno_dir) > 0:
                gt_path = os.path.join(
                    self.test_anno_dir,
                    img_metas[0]["filename"].split("/")[-1].replace(".jpg", self.ann_suffix)
                )
            else:
                gt_path = img_metas[0]["filename"].replace(
                    "images", "annotations").replace(".jpg", self.ann_suffix)
            # NOTE: process gt for different datasets
            gt = np.array(Image.open(gt_path))
            if self.ann_suffix == ".png":
                if self.reduce_zero_label:
                    gt = (gt - 1).astype(np.uint8)
                else:
                    gt = gt.astype(np.uint8)
                ignore_index = 255
            else:
                gt = gt.astype(np.int16)
                ignore_index = -1
            
            B, N, H, W = masks.shape
            assert B == 1, f"batch {B} != 1 for inference"
            grounding_mask = torch.zeros(N).bool().to(masks.device)
            unique_label = list(np.unique(gt))
            unique_label = [l for l in unique_label if l != ignore_index]
            for l in unique_label:
                grounding_mask[l] = True

            if self.oracle_inference:
                masks = self.oracle_propagation(embeds, gt_semantic_seg)
            
            # 1, n_cls, 32, 32
            masks = masks.squeeze(0)
            masks[~grounding_mask] = -100.0
            masks = masks.unsqueeze(0)
        return masks

    def oracle_propagation(self, seg_embed, seg_label):
        device = seg_embed.device
        # seg_label = torch.tensor(seg_label, dtype=torch.int64, device=device)
        B, C, H, W = seg_embed.shape
        h = seg_label.shape[-2] // self.oracle_downsample_rate
        w = seg_label.shape[-1] // self.oracle_downsample_rate
        seg_embed = resize(
            input=seg_embed,
            size=(h, w),
            mode='bilinear',
            align_corners=self.align_corners
        )
        assert self.num_classes == 150
        seg_label = resize(
            input=seg_label.float(),
            size=(h, w),
            mode='nearest'
        ).long()[0, 0]
        seg_label = seg_label - 1
        seg_label[seg_label == -1] = 255 # NOTE: hard code for convenience
        seg_embed = seg_embed.permute(0, 2, 3, 1)
        seg_label_per_image = seg_label.reshape(h * w)
        seg_embed_per_image = seg_embed.reshape(h * w, C)
        seg_embed_per_image = seg_embed_per_image / seg_embed_per_image.norm(dim=-1, keepdim=True)
        unique_label = torch.unique(seg_label_per_image)
        unique_label = unique_label[unique_label != 255]
        masks = torch.zeros((B, self.num_classes, H, W), device=device)
        for l in unique_label:
            pos_inds = (seg_label_per_image == l).nonzero(as_tuple=False)[:, 0]
            inds = torch.randperm(len(pos_inds))[:self.num_oracle_points]
            prior_inds = pos_inds[inds]
            cos_mat = seg_embed_per_image[prior_inds] @ seg_embed_per_image.T
            score_mat = cos_mat.max(dim=0).values.reshape(H, W)
            masks[0, l] = score_mat
        return masks

    @force_fp32(apply_to=('seg_mask', ))
    def losses(self, seg_mask, seg_embed, seg_feat, seg_label, img_labels=None):
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
        seg_feat = resize(
            input=seg_feat,
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
        # seg_embed = seg_embed.permute(0, 2, 3, 1).reshape(B * H * W, N)

        # prior_inds = None if len(prior_bucket) == 0 else torch.cat(prior_bucket)
        if len(prior_bucket) == 0:
            loss['loss_prior'] = torch.tensor(
                0, dtype=seg_mask.dtype, device=seg_mask.device, requires_grad=True
            )
            if self.use_pixel_embedding:
                loss['loss_emb'] = torch.tensor(
                    0, dtype=seg_mask.dtype, device=seg_mask.device, requires_grad=True
                )
            # loss['loss_structure'] = torch.tensor(
            #     0, dtype=seg_mask.dtype, device=seg_mask.device, requires_grad=True
            # )
        else:
            prior_inds = torch.cat(prior_bucket)
            # assert False, f"{seg_mask.shape, seg_label.unique(), self.ignore_index}"
            loss['loss_prior'] = self.loss_decode(
                seg_mask[prior_inds],
                seg_label[prior_inds],
                weight=None,
                ignore_index=self.ignore_index
            ) * self.prior_loss_weight
            if self.use_pixel_embedding:
                pos_bucket = prior_bucket if self.imagenet_on_gpu else None
                loss['loss_emb'] = self.loss_pix_embed(
                    seg_embed=seg_embed, seg_label=seg_label, pos_bucket=pos_bucket
                )
            # loss['loss_structure'] = self.loss_structure(
            #     seg_feat, seg_label
            # ) * self.structure_loss_weight

        acc_weight = 0.0 if self.imagenet_on_gpu else 2.0
        acc_weight = acc_weight if self.imagenet_in_batch else 1.0
        loss['acc_seg'] = accuracy(seg_mask, seg_label) * acc_weight
        return loss
    
    def _sample(self, seg_logit, seg_label, min_kept=10):
        """Sample pixels that have high loss or with low prediction confidence.

        Args:
            seg_mask (torch.Tensor): segmentation logits, shape (B, N, H, W)
            seg_label (torch.Tensor): segmentation label, shape (B, 1, H, W)

        Returns:
            pos_bucket
            prior_bucket
        """
        B, N, H, W = seg_logit.size()
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
        seg_logit = seg_logit.permute(0, 2, 3, 1).reshape(B * H * W, N)
        num_per_bucket = []
        for p in pos_bucket:
            k = int(self.prior_rate * len(p))
            if k < min_kept:
                k = min(min_kept, len(p))
            num_per_bucket.append(k)
        prior_bucket = []
        for k, p, l in zip(num_per_bucket, pos_bucket, unique_label):
            inds = seg_logit[p, int(l)].topk(k).indices
            prior_bucket.append(p[inds])
        return prior_bucket

    def _sample_imagenet(self, seg_mask, seg_embed, cam_label, img_labels):
        B, N, H, W = seg_mask.size()
        B, C, H, W = seg_embed.size()
        seg_mask = seg_mask.permute(0, 2, 3, 1).reshape(B, H * W, N).softmax(dim=-1)
        seg_embed = seg_embed.permute(0, 2, 3, 1).reshape(B, H * W, C)
        cam_label = cam_label.reshape(B, H * W)
        if self.imagenet_pseudo_label:
            prior_bucket = [
                cam.topk(int((cam > 0).sum())).indices + b * H * W for b, cam in enumerate(cam_label)
            ]
        else:
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
                    # print(f"{cam_score:.4f}, {K}, {len(inds)}, {float(cos_sim[inds].mean()):.4f}")

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


class depthwise_clipseg_conv(nn.Module):
    def __init__(self):
        super(depthwise_clipseg_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    
    def depthwise_clipseg(self, x, channels):
        x = torch.cat([self.depthwise(x[:, i].unsqueeze(1)) for i in range(channels)], dim=1)
        return x

    def forward(self, x):
        channels = x.shape[1]
        out = self.depthwise_clipseg(x, channels)
        return out


class depthwise_conv(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # support for 4D tensor with NCHW
        C, H, W = x.shape[1:]
        x = x.reshape(-1, 1, H, W)
        x = self.depthwise(x)
        x = x.view(-1, C, H, W)
        return x


class depthwise_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(depthwise_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        x = self.depthwise(x)
        if act:
            x = self.activation(x)
        return x


class bottleneck_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(bottleneck_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()


    def forward(self, x, act=True):
        sum_layer = x.max(dim=1, keepdim=True)[0]
        x = self.depthwise(x)
        x = x + sum_layer
        if act:
            x = self.activation(x)
        return x


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        activation=nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    return scratch


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x