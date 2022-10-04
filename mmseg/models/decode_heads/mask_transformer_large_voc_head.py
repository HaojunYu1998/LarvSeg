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
class MaskTransformerLargeVocHead(BaseDecodeHead):

    def __init__(
        self,
        n_cls, # for evaluation
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
        downsample_rate=8,
        temperature=1.0,
        learnable_temperature=False,
        upsample_input=1,
        # datasets
        all_cls_path="",
        mix_batch_datasets=["coco", "ade847"],
        test_dataset="ade847",
        ignore_indices=[255, -1],
        test_ignore_index=-1,
        # weakly supervised
        weakly_supervised_datasets=["ade847"],
        weakly_prior_thresh=0.8,
        weakly_min_kept=1,
        weakly_max_kept=100,
        # contrastive loss
        use_structure_loss=False,
        structure_loss_weight=1.0,
        structure_loss_thresh=0.2,
        # oracle experiment
        oracle_inference=False,
        num_oracle_points=10,
        oracle_downsample_rate=1,
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
        self.n_cls = n_cls
        self.d_model = d_model
        self.scale = d_model**-0.5
        self.prior_loss_weight = 1.0
        self.downsample_rate = downsample_rate
        self.temperature = temperature
        self.learnable_temperature = learnable_temperature
        self.upsample_input = upsample_input
        # process datasets, valid for only one dataset
        if os.path.exists(all_cls_path):
            self.all_cls = json.load(open(all_cls_path))
        else:
            self.all_cls = None
        self.mix_batch_datasets = mix_batch_datasets
        self.test_dataset = test_dataset
        self.ignore_indices = ignore_indices
        self.test_ignore_index = test_ignore_index
        # weakly supervised
        self.weakly_supervised_datasets = weakly_supervised_datasets
        self.weakly_prior_thresh = weakly_prior_thresh
        self.weakly_min_kept = weakly_min_kept
        self.weakly_max_kept = weakly_max_kept
        # contrastive loss
        self.use_structure_loss = use_structure_loss
        self.structure_loss_weight = structure_loss_weight
        self.structure_loss_thresh = structure_loss_thresh
        # oracle experiment
        self.oracle_inference = oracle_inference
        self.num_oracle_points = num_oracle_points
        self.oracle_downsample_rate = oracle_downsample_rate

        self.proj_dec = nn.Linear(d_encoder, d_model)
        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.gamma = nn.Parameter(torch.ones([]))
        self.beta = nn.Parameter(torch.zeros([]))
        # NOTE: cosine classifier
        if self.all_cls is not None:
            self.cls_emb = nn.Parameter(
                # all categories for training
                self.scale * torch.randn(len(self.all_cls), d_model)
            )
        else:
            self.cls_emb = nn.Parameter(
                self.scale * torch.randn(self.num_classes, d_model)
            )
        if self.learnable_temperature:
            self.logit_scale = nn.Parameter((torch.ones([]) * np.log(1 / 0.07)).exp())

    def init_weights(self):
        self.apply(init_weights)

    def _update(self, training):
        rank, _ = get_dist_info()
        if training:
            self.dataset_on_gpu = self.mix_batch_datasets[rank % len(self.mix_batch_datasets)]
            self.ignore_index = self.ignore_indices[rank % len(self.mix_batch_datasets)]
        else:
            self.dataset_on_gpu = self.test_dataset
            self.ignore_index = self.test_ignore_index
        
        if self.dataset_on_gpu == "coco171":
            from mmseg.datasets.coco_stuff import COCOStuffDataset
            cls_name = COCOStuffDataset.CLASSES
        elif self.dataset_on_gpu == "ade150":
            from mmseg.datasets.ade import ADE20KDataset
            cls_name = ADE20KDataset.CLASSES
        elif self.dataset_on_gpu == "ade847":
            from mmseg.datasets.ade import ADE20KFULLDataset
            cls_name = ADE20KFULLDataset.CLASSES

        if self.all_cls is not None:
            self.cls_index = [self.all_cls.index(name) for name in cls_name]
        else:
            self.cls_index = list(range(len(cls_name)))

    def forward(self, x, img_metas, img_labels=None):
        x = self._transform_inputs(x)
        if self.upsample_input > 1:
            x = F.interpolate(
                x, scale_factor=self.upsample_input, mode="bilinear", align_corners=self.align_corners
            )
        cls_emb = self.cls_emb[self.cls_index]
        cls_emb = cls_emb.expand(x.size(0), -1, -1)
        B, C, H, W = x.size()
        feats = x.clone()
        x = x.view(B, C, -1).permute(0, 2, 1)

        x = self.proj_dec(x)
        patches, cls_seg_feat = x, cls_emb
        patches = patches @ self.proj_patch
        # B, HW, C
        patches = patches / patches.norm(dim=-1, keepdim=True)
        # B, N, C
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        if self.learnable_temperature:
            masks = self.logit_scale * patches @ cls_seg_feat.transpose(1, 2)
        else:
            masks = patches @ cls_seg_feat.transpose(1, 2) / self.temperature
        if self.training:
            masks = (
                (masks - torch.mean(masks, dim=-1, keepdim=True))
                / torch.sqrt(torch.var(masks, dim=-1, keepdim=True, unbiased=False) + 1e-5)
            ) * self.gamma + self.beta
        B, HW, N = masks.size()

        masks = masks.view(B, H, W, N).permute(0, 3, 1, 2)
        embeds = patches.clone()
        embeds = embeds.view(B, H, W, -1).permute(0, 3, 1, 2)
        return masks, embeds, feats

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        self._update(training=True)
        img_labels = None
        masks, embeds, feats = self.forward(inputs, img_metas)
        losses = self.losses(masks, embeds, feats, gt_semantic_seg, img_labels)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg=None):
        self._update(training=False)
        masks, embeds, feats = self.forward(inputs, img_metas)
        if self.oracle_inference:
            assert gt_semantic_seg is not None
            masks = self.oracle_propagation(embeds, gt_semantic_seg)
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
        seg_label[seg_label == -1] = 255 # NOTE: hard code for convenience (ADE-150)
        seg_embed = seg_embed.permute(0, 2, 3, 1)
        seg_label_per_image = seg_label.reshape(h * w)
        seg_embed_per_image = seg_embed.reshape(h * w, C)
        seg_embed_per_image = seg_embed_per_image / seg_embed_per_image.norm(dim=-1, keepdim=True)
        unique_label = torch.unique(seg_label_per_image)
        unique_label = unique_label[unique_label != 255]
        masks = torch.zeros((B, self.num_classes, h, w), device=device)
        for l in unique_label:
            pos_inds = (seg_label_per_image == l).nonzero(as_tuple=False)[:, 0]
            inds = torch.randperm(len(pos_inds))[:self.num_oracle_points]
            prior_inds = pos_inds[inds]
            cos_mat = seg_embed_per_image[prior_inds] @ seg_embed_per_image.T
            score_mat = cos_mat.max(dim=0).values.reshape(h, w)
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
        if self.dataset_on_gpu in self.weakly_supervised_datasets:
            prior_mask, prior_label = self._weak_sample(seg_mask, seg_label)
        else:
            prior_mask, prior_label = self._sample(seg_mask, seg_label)

        seg_label = seg_label.reshape(B * H * W)
        seg_mask = seg_mask.permute(0, 2, 3, 1).reshape(B * H * W, N)

        if prior_mask is None:
            loss['loss_prior'] = torch.tensor(
                0, dtype=seg_mask.dtype, device=seg_mask.device, requires_grad=True
            )
            if self.use_structure_loss:
                loss['loss_structure'] = torch.tensor(
                    0, dtype=seg_mask.dtype, device=seg_mask.device, requires_grad=True
                )
        else:
            assert prior_label is not None
            loss['loss_prior'] = self.loss_decode(
                prior_mask, 
                prior_label,
                weight=None,
                ignore_index=self.ignore_index
            ) * self.prior_loss_weight
            if self.use_structure_loss:
                loss['loss_structure'] = self.loss_structure(
                    seg_feat, seg_label
                ) * self.structure_loss_weight

        acc_weight = 1.0
        loss['acc_seg'] = accuracy(seg_mask, seg_label) * acc_weight
        return loss
    
    def _sample(self, seg_mask, seg_label, min_kept=1):
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
            k = len(p)
            if k < min_kept:
                k = min(min_kept, len(p))
            num_per_bucket.append(k)
        prior_bucket = []
        for k, p, l in zip(num_per_bucket, pos_bucket, unique_label):
            inds = seg_mask[p, int(l)].topk(k).indices
            prior_bucket.append(p[inds])
        # don't know what happened to cause this
        if len(prior_bucket) <= 0:
            return None, None
        prior_inds = torch.cat(prior_bucket)
        seg_label = seg_label.reshape(B * H * W)
        # seg_mask = seg_mask.permute(0, 2, 3, 1).reshape(B * H * W, N)
        prior_mask = seg_mask[prior_inds]
        prior_label = seg_label[prior_inds]
        return prior_mask, prior_label

    def _weak_sample(self, seg_mask, seg_label):
        """
        for each image, smaple each category separately
        i.e. a pixel could be assigned to more than one categories
        """
        B, N, H, W = seg_mask.size()
        prior_mask, prior_label = [], []
        for mask, label in zip(seg_mask, seg_label):
            mask = mask.reshape(N, H * W).permute(1, 0)
            label = label.reshape(H * W)
            unique_label = torch.unique(label)
            unique_label = unique_label[unique_label != self.ignore_index]
            for l in unique_label:
                inds = (mask[:, l] > self.weakly_prior_thresh).nonzero(as_tuple=False).flatten()
                if inds.numel() < self.weakly_min_kept:
                    inds = mask[:, l].topk(self.weakly_min_kept).indices
                elif inds.numel() > self.weakly_max_kept:
                    inds = mask[:, l].topk(self.weakly_max_kept).indices
                prior_mask.append(mask[inds])
                prior_label.append(label[inds])
        prior_mask = torch.cat(prior_mask, dim=0)
        prior_label = torch.cat(prior_label, dim=0)
        # assert False, f"{prior_mask.shape}"
        return prior_mask, prior_label

    def loss_structure(self, seg_feat, seg_label):
        if self.dataset_on_gpu in self.weakly_supervised_datasets:
            return torch.tensor(
                0, dtype=seg_feat.dtype, device=seg_feat.device, requires_grad=True
            )
        B, C, H, W = seg_feat.size()
        seg_feat = seg_feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
        seg_label = seg_label.reshape(B * H * W)
        unique_label = torch.unique(seg_label)
        pos_bucket = [
            torch.nonzero(seg_label == l)[:, 0]
            for l in unique_label
            if l != self.ignore_index
        ]
        if len(pos_bucket) == 0:
            return seg_feat[seg_label != self.ignore_index].sum()
        pos_inds = self._sample_feat(pos_bucket)
        sample_cls = torch.cat([
            seg_label[[i]] for i in pos_inds], dim=0).to(seg_feat.device)
        sample_feat = torch.cat([
            seg_feat[i] for i in pos_inds], dim=0)
        loss = self.loss_similarity(sample_feat, sample_cls)
        return loss

    def loss_similarity(self, feat, label):
        """Compute the similarity loss
        Args:
            embedding (torch.Tensor): [512, C]
            label (torch.Tensor): [512]
        """
        feat = feat / feat.norm(dim=-1, keepdim=True)
        cos_sim = feat @ feat.T  # [B,B]
        label_sim = (label[None, :] == label[:, None]).int().float()
        valid_mask = torch.ones_like(cos_sim)
        valid_mask[
            torch.arange(len(valid_mask)).to(valid_mask.device),
            torch.arange(len(valid_mask)).to(valid_mask.device),
        ] = 0
        cos_sim = cos_sim[valid_mask.bool()]
        label_sim = label_sim[valid_mask.bool()]
        # NOTE: for negative samples, don't add loss if they are lower than the thresh
        _mask = (
            (cos_sim > self.structure_loss_thresh) | (label_sim == 1)
        )
        cos_sim = cos_sim[_mask]
        label_sim = label_sim[_mask]
        return torch.pow(cos_sim - label_sim, 2)
    
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