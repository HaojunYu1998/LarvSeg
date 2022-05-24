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
class MaskTransformerStructureHead(BaseDecodeHead):

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
        imagenet_pseudo_label=False,
        imagenet_sample_class_num=0,
        imagenet_cam_thresh=0,
        pseudo_label_thresh=0.0,
        propagation_loss_weight=0.0,
        structure_loss_weight=0.0,
        structure_loss_method="margin",
        structure_queue_len=32,
        structure_hard_smapling=False,
        structure_per_image=False,
        structure_temperature=0.2,
        structure_margin=0.3,
        structure_min_value=0.0,
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
        self.structure_loss_method = structure_loss_method
        self.structure_queue_len = structure_queue_len
        self.structure_hard_smapling = structure_hard_smapling
        self.structure_temperature = structure_temperature
        self.structure_per_image = structure_per_image
        self.structure_margin = structure_margin
        self.structure_min_value = structure_min_value
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
        self.imagenet_on_gpu_ = "in11k" in self.cls_emb_path if self.training \
                          else "in11k" in self.cls_emb_path_test
        self.imagenet_on_gpu = self.imagenet_on_gpu_ or self.imagenet_on_gpu
        # assert self.imagenet_on_gpu is True, f"{self.cls_emb_path_test}, {self.training}"
        # print(rank, self.imagenet)
        if self.imagenet_on_gpu:
            self.prior_loss_weight = imagenet_prior_loss_weight
            if imagenet_class_path is not None:
                with open(imagenet_class_path, "r") as f:
                    in21k_id_name_dict = json.load(f)
                    # in21k_names = list(in21k_id_name_dict.values())
                    self.in21k_ids = list(in21k_id_name_dict.keys())
        self.label_cos_sim = None

        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale *
                                       torch.randn(d_model, d_model))
        # self.proj_classes = nn.Parameter(self.scale *
        #                                  torch.randn(d_model, d_model))
        
        self.decoder_norm = nn.LayerNorm(d_model)
        self.gamma = nn.Parameter(torch.ones([]))
        self.beta = nn.Parameter(torch.zeros([]))
        # self.mask_norm = nn.LayerNorm(n_cls)

        if self.structure_loss_method == "region":
            self.structure_criterion = nn.CrossEntropyLoss()
        self.set_structure_queue = True

    def init_weights(self):
        # if self.call_init_weight:
        self.apply(init_weights)
        # if not self.cls_emb_from_backbone:
        #     trunc_normal_(self.cls_emb, std=0.02)

    def forward(self, x, img_metas):
        if self.cls_emb_from_backbone:
            x, cls_emb = x
        else:
            cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = self._transform_inputs(x)
        cls_emb = cls_emb.to(x.device)

        B, C, H, W = x.size()
        
        x = x.view(B, C, -1).permute(0, 2, 1)

        x = self.proj_dec(x)
        feats = x.clone()
        feats = feats.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
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
            cls_emb = torch.load(self.cls_emb_path, map_location="cpu")
            if isinstance(cls_emb, dict):
                self.in21k_ids_all = list(cls_emb.keys())
                cls_emb = torch.stack(list(cls_emb.values()))
                self.cls_emb_all = cls_emb.clone()
            self.structure_num_classes = len(cls_emb)
            if self.structure_loss_method == "region" and self.set_structure_queue:
                for i in range(self.structure_num_classes):
                    self.register_buffer(
                        "queue"+str(i), torch.randn(self.d_encoder, self.structure_queue_len)
                    )
                    self.register_buffer(
                        "ptr"+str(i), torch.zeros(1, dtype=torch.long)
                    )
                    exec("self.queue"+str(i) + '=' + 'nn.functional.normalize(' + "self.queue"+str(i) + ',dim=0).cuda()')
                self.set_structure_queue = False
            self.cls_emb = cls_emb
            self.cls_emb.requires_grad = False
            self.loaded_cls_emb_test = False
            self.loaded_cls_emb_train = True

        img_labels = None
        if self.imagenet_on_gpu: # if self.imagenet_sample_class_num > 0:
            img_names = [meta['ori_filename'].split("/")[-1] for meta in img_metas]
            img_ids = [name[:name.find("_")] for name in img_names]
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
                self.sampled_idx_to_ori_inds = {
                    i: self.in21k_ids_all.index(img_id)
                    for i, img_id in enumerate(self.in21k_ids)
                }
            img_labels = [self.in21k_ids.index(img_id) for img_id in img_ids]

        masks, embeds, feats = self.forward(inputs, img_metas)

        losses = self.losses(masks, embeds, feats, gt_semantic_seg, img_labels)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        if not self.loaded_cls_emb_test:
            self.cls_emb = torch.load(
                self.cls_emb_path_test, map_location="cpu")
            self.loaded_cls_emb_test = True
            self.loaded_cls_emb_train = False
        
        masks, _, _ = self.forward(inputs, img_metas)

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
        seg_feat = seg_feat.permute(0, 2, 3, 1).reshape(B * H * W, -1)

        # prior_inds = None if len(prior_bucket) == 0 else torch.cat(prior_bucket)
        if len(prior_bucket) == 0:
            loss['loss_prior'] = torch.tensor(
                0, dtype=seg_mask.dtype, device=seg_mask.device, requires_grad=True
            )
            loss['loss_structure'] = torch.tensor(
                0, dtype=seg_mask.dtype, device=seg_mask.device, requires_grad=True
            )
        else:
            prior_inds = torch.cat(prior_bucket)
            assert prior_inds.numel() > 0
            loss['loss_prior'] = self.loss_decode(
                seg_mask[prior_inds],
                seg_label[prior_inds],
                weight=None,
                ignore_index=self.ignore_index
            ) * self.prior_loss_weight
            if not self.structure_per_image:
                loss['loss_structure'] = self.loss_structure(
                    seg_feat[prior_inds],
                    seg_mask[prior_inds],
                    seg_label[prior_inds],
                ) * self.structure_loss_weight
            else:
                seg_label = seg_label.reshape(B, H * W)
                seg_mask = seg_mask.reshape(B, H * W, N)
                seg_feat = seg_feat.reshape(B, H * W, -1)
                loss_structure = torch.tensor(
                    0, dtype=seg_mask.dtype, device=seg_mask.device, requires_grad=True
                )
                for b in range(B):
                    prior_inds_per_image = prior_inds[
                        (prior_inds >= b * H * W) & 
                        (prior_inds < (b + 1) * H * W)
                    ]
                    loss_structure += self.loss_structure(
                        seg_feat[b][prior_inds_per_image],
                        seg_mask[b][prior_inds_per_image],
                        seg_label[b][prior_inds_per_image],
                    )
                loss['loss_structure'] = loss_structure / B * self.structure_loss_weight

        acc_weight = 0.0 if self.imagenet_on_gpu else 2.0
        acc_weight = acc_weight if self.imagenet_in_batch else 1.0
        loss['acc_seg'] = accuracy(seg_mask, seg_label) * acc_weight
        return loss

    def loss_structure(self, seg_feat, seg_mask, seg_label):
        """
        seg_feat: (N_prior, C)
        seg_mask: (N_prior, N)
        seg_label: (N_prior, )
        """
        unique_label = torch.unique(seg_label)
        pos_bucket = [
            torch.nonzero(seg_label == l)[:, 0]
            for l in unique_label
            if l != self.ignore_index
        ]
        if len(pos_bucket) == 0:
            return seg_feat[seg_label != self.ignore_index].sum()

        pos_inds = self._sample_feat(pos_bucket)
        sample_feat = torch.cat([
            seg_feat[i] for i in pos_inds], dim=0)

        if self.structure_loss_method == "region":
            region_feat, unique_label = self.construct_region(
                seg_feat, seg_mask, seg_label
            )
            unique_label = unique_label[unique_label != self.ignore_index]
            loss = self.similarity3(region_feat, unique_label)
        elif self.structure_loss_method == "contrastive":
            sample_cls = torch.cat(
                [torch.Tensor([i for _ in range(len(p))]) for i, p in enumerate(pos_inds)],
                dim=0,
            ).to(sample_feat.device)
            loss = self.similarity2(sample_feat, sample_cls)
        elif self.structure_loss_method == "margin":
            sample_cls = torch.cat([
                seg_label[[i]] for i in pos_inds], dim=0
            ).to(seg_feat.device).long()
            loss = self.similarity1(sample_feat, sample_cls)
        return loss

    def similarity1(self, feat, label):
        """Compute the similarity loss
        Args:
            feat (torch.Tensor): [B,C]
            label (torch.Tensor): [B]
        """
        feat = feat / feat.norm(dim=-1, keepdim=True)
        cos_sim = feat @ feat.T  # [B,B]
        cls_emb = self.cls_emb.to(feat.device)
        cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
        # print(label)
        label_sim = cls_emb[label] @ cls_emb[label].T
        # print(cos_sim, label_sim)
        pos_mask = (label[:, None] == label[None, :]).type(feat.dtype)  # [B,B]
        neg_mask = 1 - pos_mask
        # remove self-to-self sim
        pos_mask[
            torch.arange(len(pos_mask)).to(pos_mask.device),
            torch.arange(len(pos_mask)).to(pos_mask.device),
        ] = 0
        pos_mask = pos_mask.flatten().bool()
        neg_mask = neg_mask.flatten().bool()
        cos_sim = cos_sim.flatten()
        label_sim = label_sim.flatten()
        label_sim = label_sim[neg_mask] - self.structure_margin
        label_sim = label_sim.clamp(min=self.structure_min_value)
        pos_sim = cos_sim[pos_mask] - 1
        neg_sim = cos_sim[neg_mask]
        neg_sim = neg_sim[neg_sim >= label_sim]
        if pos_sim.numel() > 0 and neg_sim.numel() > 0:
            loss = torch.pow(pos_sim, 2).mean() + torch.pow(neg_sim, 2).mean()
        else:
            loss = torch.tensor(
                0, dtype=feat.dtype, device=feat.device, requires_grad=True
            )
        return loss

    def similarity2(self, feat, label, temperature=0.02):
        """Compute the similarity loss
        Args:
            feat (torch.Tensor): [B,C]
            label (torch.Tensor): [B]
        """
        feat = feat / feat.norm(dim=-1, keepdim=True)
        cos_sim = feat @ feat.T  # [B,B]
        exp_sim = torch.exp(cos_sim / temperature)
        pos_mask = (label[:, None] == label[None, :]).type(exp_sim.dtype)  # [B,B]
        neg_mask = 1 - pos_mask
        # remove self-to-self sim
        pos_mask[
            torch.arange(len(pos_mask)).to(pos_mask.device),
            torch.arange(len(pos_mask)).to(pos_mask.device),
        ] = 0

        neg_exp_sim_sum = (exp_sim * neg_mask).sum(dim=-1, keepdim=True)
        prob = exp_sim / (exp_sim + neg_exp_sim_sum).clamp(min=1e-8)
        # select positive pair
        pos_prob = prob[pos_mask == 1]
        loss = -torch.log(pos_prob + 1e-8).mean()
        return loss

    def similarity3(self, feat, label):
        """
        feat: (N_region, C)
        label: (N_region)
        """
        feat = feat / feat.norm(dim=-1, keepdim=True)
        contrast_loss = 0
        for q, l in zip(feat, label):
            if self.imagenet_on_gpu and self.imagenet_sample_class_num > 0:
                l = self.sampled_idx_to_ori_inds[int(l)]
            l = int(l)
            # k = eval("self.queue"+str(l)).clone().detach()
            # l_pos = q[None] @ k # (1, N_queue)
            l_pos = q.unsqueeze(1)*eval("self.queue"+str(l)).clone().detach()  #256, N1
            all_ind = [m for m in range(self.structure_num_classes)]
            tmp = all_ind.copy()
            tmp.remove(l)
            # l_neg = []
            l_neg = 0
            for neg_l in tmp:
                # k = eval("self.queue"+str(neg_l)).clone().detach()
                # l_neg.append(q[None] @ k)
                l_neg += q.unsqueeze(1)*eval("self.queue"+str(neg_l)).clone().detach()
            # (N_neg, N_queue)
            # l_neg = torch.cat(l_neg, dim=0)
            contrast_loss += self._compute_contrast_loss(l_pos, l_neg)

        for q, l in zip(feat, label):
            if self.imagenet_on_gpu and self.imagenet_sample_class_num > 0:
                l = self.sampled_idx_to_ori_inds[int(l)]
            l = int(l)
            self._dequeue_and_enqueue(q, l)
        return contrast_loss #/ max(1, len(label))
        # return contrast_loss

    def construct_region(self, feat, mask, label):
        unique_label = torch.unique(label)
        unique_label = unique_label[unique_label != self.ignore_index]
        if self.structure_hard_smapling:
            pred = mask.softmax(dim=-1)
            values, indices = pred.max(dim=-1)
            values = values.flatten()
            assert (values >= 0).all() and (values <= 1).all()
            indices = indices.flatten()
            region_feat = []
            for l in unique_label:
                easy_mask = (indices == l) & (label == l)
                hard_mask = (indices != l) & (label == l)
                # print(f"{int(easy_mask.sum())}, {int(hard_mask.sum())}")
                feat_l = torch.cat(
                    [(1 - values[easy_mask].unsqueeze(1)) * feat[easy_mask], feat[hard_mask]], dim=0
                ).mean(dim=0)
                region_feat.append(feat_l)
            region_feat = torch.stack(region_feat, dim=0)
        else:
            region_feat = torch.stack([
                label[label==l].mean(dim=0)
                for l in unique_label
            ]) # (N_region, C)
        return region_feat, unique_label

    def _compute_contrast_loss(self, l_pos, l_neg):
        # """
        # l_pos: (1, N_queue)
        # l_neg: (N_neg, N_queue)
        # """
        # N_queue = l_pos.size(1)
        # logits = torch.cat((l_pos, l_neg), dim=0)
        # logits = logits.transpose(0, 1) # (N_queue, 1+N_neg)
        # logits /= self.structure_temperature
        # labels = torch.zeros((N_queue, ),dtype=torch.long).cuda()
        # return self.structure_criterion(logits, labels)
        N = l_pos.size(0)
        logits = torch.cat((l_pos, l_neg), dim=1)
        logits /= self.structure_temperature
        labels = torch.zeros((N,), dtype=torch.long).cuda()
        return self.structure_criterion(logits, labels)

    def _dequeue_and_enqueue(self, feat, label): #, bs):
        ptr = int(eval("self.ptr"+str(label)))
        eval("self.queue"+str(label))[:, ptr] = feat
        ptr = (ptr + 1) % self.structure_queue_len
        eval("self.ptr"+str(label))[0] = ptr
        # print(ptr)

    def _sample_feat(self, buckets, total_sample_num=512):
        """Sample points from each buckets
        Args:
            num_per_buckets (list): number of points in each class
        """
        num_per_buckets = [len(p) for p in buckets]
        sample_per_bucket = [
            total_sample_num // len(buckets)
            for _ in range(len(num_per_buckets))
        ]
        
        if len(sample_per_bucket) > 1:
            sample_per_bucket[-1] = total_sample_num - sum(sample_per_bucket[:-1])
        else:
            sample_per_bucket[0] = total_sample_num
        samples = [
            p[
                torch.from_numpy(
                    np.random.choice(len(p), sample_per_bucket[i], replace=True)
                ).to(p.device)
            ]
            for i, p in enumerate(buckets)
        ]
        return samples

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
