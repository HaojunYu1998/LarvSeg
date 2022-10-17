import copy
import json
import math
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import unique
from genericpath import exists
from importlib_metadata import requires
from mmcv.runner import force_fp32, get_dist_info
from PIL import Image
from random import sample
from timm.models.layers import DropPath
from torch.nn.init import trunc_normal_

from mmseg.ops import resize
from ..builder import HEADS
from ..losses.accuracy import accuracy
from .decode_head import BaseDecodeHead


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
        upsample_input=1,
        # datasets
        all_cls_path="",
        mix_batch_datasets=["coco", "ade847"],
        test_dataset="ade847",
        ignore_indices=[255, -1],
        test_ignore_index=-1,
        # separate heads
        use_attn_head=False,
        # weakly supervised
        weakly_supervised_datasets=["ade847"],
        weakly_seed_thresh=0.8,
        weakly_min_kept=1,
        weakly_max_kept=100,
        weakly_seed_loss_weight=1.0,
        # contrastive loss
        use_structure_loss=False,
        structure_loss_weight=1.0,
        structure_loss_thresh=0.2,
        structure_loss_no_negative=False,
        # contrastive loss for class embedding
        cls_structure_loss_weight=1.0,
        cls_structure_loss_thresh=0.2,
        # oracle experiment
        oracle_inference=False,
        num_oracle_points=10,
        oracle_downsample_rate=1,
        # visualization
        visualize_seed=False,
        visualize_out_dir="",
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
        self.downsample_rate = downsample_rate
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
        self.seed_loss_weight = 1.0
        self.use_attn_head = use_attn_head
        # weakly supervised
        self.weakly_supervised_datasets = weakly_supervised_datasets
        self.weakly_seed_thresh = weakly_seed_thresh
        self.weakly_min_kept = weakly_min_kept
        self.weakly_max_kept = weakly_max_kept
        self.weakly_seed_loss_weight = weakly_seed_loss_weight
        # contrastive loss
        self.use_structure_loss = use_structure_loss
        self.structure_loss_weight = structure_loss_weight
        self.structure_loss_thresh = structure_loss_thresh
        self.structure_loss_no_negative = structure_loss_no_negative
        # contrastive loss for classification embedding
        self.cls_structure_loss_weight = cls_structure_loss_weight
        self.cls_structure_loss_thresh = cls_structure_loss_thresh
        # oracle experiment
        self.oracle_inference = oracle_inference
        self.num_oracle_points = num_oracle_points
        self.oracle_downsample_rate = oracle_downsample_rate
        # visualization
        self.visualize_seed = visualize_seed
        self.visualize_out_dir = visualize_out_dir

        self.proj_dec = nn.Linear(d_encoder, d_model)
        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        # attention head
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList([
            Block(dim=d_model, heads=n_heads, mlp_dim=d_ff, dropout=dropout, drop_path=dpr[i])
            for i in range(n_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)
        # cosine classifier
        self.gamma = nn.Parameter(torch.ones([]))
        self.beta = nn.Parameter(torch.zeros([]))
        num_classes = self.num_classes if self.all_cls is None else len(self.all_cls)
        self.cls_emb = nn.Parameter(torch.randn(num_classes, d_model))

    def init_weights(self):
        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    def _update(self, training):
        rank, _ = get_dist_info()
        if training:
            self.dataset_on_gpu = self.mix_batch_datasets[rank % len(self.mix_batch_datasets)]
            self.ignore_index = self.ignore_indices[rank % len(self.mix_batch_datasets)]
            if self.dataset_on_gpu in self.weakly_supervised_datasets:
                self.seed_loss_weight = self.weakly_seed_loss_weight
        else:
            self.dataset_on_gpu = self.test_dataset
            self.ignore_index = self.test_ignore_index
        
        if self.dataset_on_gpu == "coco171":
            from mmseg.datasets.coco_stuff import COCOStuffDataset
            cls_name = COCOStuffDataset.CLASSES
            cls_name = [x.split("-")[0] for x in cls_name]
        elif self.dataset_on_gpu == "ade150":
            from mmseg.datasets.ade import ADE20KDataset
            cls_name = ADE20KDataset.CLASSES
        elif self.dataset_on_gpu == "ade130":
            from mmseg.datasets.ade import ADE20K130Dataset
            cls_name = ADE20K130Dataset.CLASSES130
        elif self.dataset_on_gpu == "ade847":
            from mmseg.datasets.ade import ADE20KFULLDataset
            cls_name = ADE20KFULLDataset.CLASSES
        elif self.dataset_on_gpu == "ade585":
            from mmseg.datasets.ade import ADE20K585Dataset
            cls_name = ADE20K585Dataset.CLASSES585
        elif self.dataset_on_gpu == "in130":
            from mmseg.datasets.imagenet import ImageNet130
            cls_name = ImageNet130.CLASSES
        elif self.dataset_on_gpu == "in585":
            from mmseg.datasets.imagenet import ImageNet585
            cls_name = ImageNet585.CLASSES
        else:
            raise NotImplementedError(f"{self.dataset_on_gpu} is not supported")

        self.cls_name = cls_name
        if self.all_cls is not None:
            self.cls_index = [self.all_cls.index(name) for name in cls_name]
        else:
            self.cls_index = list(range(len(cls_name)))

    def forward(self, x, img_metas, img_labels=None):
        x = self._transform_inputs(x)
        B, C, H, W = x.size()
        x = x.view(B, C, -1).permute(0, 2, 1)
        x = self.proj_dec(x)
        cls_emb = self.cls_emb[self.cls_index]
        cls_emb = cls_emb.expand(x.size(0), -1, -1)
        for blk in self.blocks:
            x = blk(x)
        patches = self.decoder_norm(x)
        patches = patches @ self.proj_patch
        cls_emb = cls_emb @ self.proj_classes
        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
        masks = patches @ cls_emb.transpose(1, 2)
        scores = masks.clone().detach()
        embeds = patches.clone()
        if self.training:
            masks = (
                (masks - torch.mean(masks, dim=-1, keepdim=True))
                / torch.sqrt(torch.var(masks, dim=-1, keepdim=True, unbiased=False) + 1e-5)
            ) * self.gamma + self.beta
        B, HW, N = masks.size()
        masks = masks.view(B, H, W, N).permute(0, 3, 1, 2)
        embeds = embeds.view(B, H, W, -1).permute(0, 3, 1, 2)
        scores = scores.view(B, H, W, N).permute(0, 3, 1, 2)
        return masks, embeds, scores

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        self._update(training=True)
        img_labels = None
        masks, embeds, scores = self.forward(inputs, img_metas)
        losses = self.losses(masks, embeds, scores, gt_semantic_seg, img_labels)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg=None, img=None):
        self._update(training=False)
        masks, embeds, _ = self.forward(inputs, img_metas)
        if self.oracle_inference:
            assert gt_semantic_seg is not None
            masks = self.oracle_propagation(embeds, gt_semantic_seg)
        return masks

    @force_fp32(apply_to=('seg_mask', ))
    def losses(self, seg_mask, seg_embed, seg_score, seg_label, img_labels=None):
        """Compute segmentation loss."""
        loss = dict()
        h = seg_label.shape[-2] // self.downsample_rate
        w = seg_label.shape[-1] // self.downsample_rate
        seg_mask = resize(
            seg_mask, size=(h, w), mode='bilinear', align_corners=self.align_corners
        )
        seg_score = resize(
            seg_score, size=(h, w), mode='bilinear', align_corners=self.align_corners
        )
        seg_embed = resize(
            seg_embed, size=(h, w), mode='bilinear', align_corners=self.align_corners
        )
        seg_label = resize(
            seg_label.float(), size=(h, w), mode='nearest'
        ).long()

        # classification task
        B, N, H, W = seg_mask.shape
        if self.dataset_on_gpu in self.weakly_supervised_datasets:
            seed_mask, seed_label = self._weak_sample(seg_mask, seg_score, seg_label)
        else:
            seed_mask, seed_label = self._sample(seg_mask, seg_label)
        # no valid label
        if seed_mask is None:
            loss['loss_seed'] = seg_mask.sum() * 0.0
        else:
            assert seed_label is not None
            loss['loss_seed'] = self.loss_decode(
                seed_mask, 
                seed_label,
                weight=None,
                ignore_index=self.ignore_index
            ) * self.seed_loss_weight
        # log accuracy
        loss['acc_seg'] = self.log_accuracy(seg_mask, seg_label)

        # localization task
        if self.use_structure_loss:
            loss['loss_structure'] = self.loss_structure(
                seg_embed, seg_label
            ) * self.structure_loss_weight
        return loss
    
    def log_accuracy(self, seg_mask, seg_label):
        B, N, H, W = seg_mask.shape
        seg_label = seg_label.flatten()
        seg_mask = seg_mask.permute(0, 2, 3, 1).reshape(B * H * W, N)
        imagenet_in_batch = any(["in" in x for x in self.mix_batch_datasets])
        acc_weight = 0.0 if "in" in self.dataset_on_gpu else 2.0
        acc_weight = acc_weight if imagenet_in_batch else 1.0
        return accuracy(seg_mask, seg_label)
    
    def _sample(self, seg_mask, seg_label, min_kept=1):
        """
        Args:
            seg_mask (torch.Tensor): segmentation logits, shape (B, N, H, W)
            seg_label (torch.Tensor): segmentation label, shape (B, 1, H, W)

        Returns:
            pos_bucket
            seed_bucket
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
            return None, None
        seg_mask = seg_mask.permute(0, 2, 3, 1).reshape(B * H * W, N)
        num_per_bucket = []
        for p in pos_bucket:
            k = len(p)
            if k < min_kept:
                k = min(min_kept, len(p))
            num_per_bucket.append(k)
        seed_bucket = []
        for k, p, l in zip(num_per_bucket, pos_bucket, unique_label):
            inds = seg_mask[p, int(l)].topk(k).indices
            seed_bucket.append(p[inds])
        # don't know what happened to cause this
        if len(seed_bucket) == 0:
            return None, None
        seed_inds = torch.cat(seed_bucket)
        seg_label = seg_label.reshape(B * H * W)
        seed_mask = seg_mask[seed_inds]
        seed_label = seg_label[seed_inds]
        return seed_mask, seed_label

    def _weak_sample(self, seg_mask, seg_score, seg_label):
        """
        for each image, smaple each category separately
        i.e. a pixel could be assigned to more than one categories
        """
        B, N, H, W = seg_mask.size()
        seed_mask, seed_label = [], []
        for mask, score, label in zip(seg_mask, seg_score, seg_label):
            mask = mask.reshape(N, H * W).permute(1, 0)
            label = label.reshape(H * W)
            score = score.reshape(N, H * W).permute(1, 0)
            unique_label = torch.unique(label)
            unique_label = unique_label[unique_label != self.ignore_index]
            for l in unique_label:
                inds = (score[:, l] > self.weakly_seed_thresh).nonzero(as_tuple=False).flatten()
                if inds.numel() < self.weakly_min_kept:
                    inds = score[:, l].topk(self.weakly_min_kept).indices
                elif inds.numel() > self.weakly_max_kept:
                    inds = score[:, l].topk(self.weakly_max_kept).indices
                seed_mask.append(mask[inds])
                seed_label.append(torch.ones_like(label[inds]) * l)
        seed_mask = torch.cat(seed_mask, dim=0)
        seed_label = torch.cat(seed_label, dim=0)
        return seed_mask, seed_label
    
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
        loss = self.loss_similarity(sample_feat, sample_cls, self.structure_loss_thresh)
        return loss

    def loss_similarity(self, feat, label, thresh):
        """Compute the similarity loss
        Args:
            embedding (torch.Tensor): [N, C]
            label (torch.Tensor): [N]
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
        if self.structure_loss_no_negative:
            _mask = label_sim == 1
        else:
            # NOTE: for negative samples, don't add loss if they are lower than the thresh
            _mask = (
                (cos_sim > thresh) | (label_sim == 1)
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
        # assert self.num_classes == 150
        seg_label = resize(
            input=seg_label.float(),
            size=(h, w),
            mode='nearest'
        ).long()[0, 0]
        if self.dataset_on_gpu == "ade150":
            seg_label = seg_label - 1
            seg_label[seg_label == -1] = self.ignore_index
        seg_embed = seg_embed.permute(0, 2, 3, 1)
        seg_label_per_image = seg_label.reshape(h * w)
        seg_embed_per_image = seg_embed.reshape(h * w, C)
        seg_embed_per_image = seg_embed_per_image / seg_embed_per_image.norm(dim=-1, keepdim=True)
        unique_label = torch.unique(seg_label_per_image)
        unique_label = unique_label[unique_label != self.ignore_index]
        masks = torch.zeros((B, self.num_classes, h, w), device=device)
        for l in unique_label:
            pos_inds = (seg_label_per_image == l).nonzero(as_tuple=False)[:, 0]
            inds = torch.randperm(len(pos_inds))[:self.num_oracle_points]
            seed_inds = pos_inds[inds]
            cos_mat = seg_embed_per_image[seed_inds] @ seg_embed_per_image.T
            score_mat = cos_mat.max(dim=0).values.reshape(h, w)
            masks[0, l] = score_mat
        return masks

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

    def forward(self, x):
        B, _, C = x.shape
        # B, head, N, C // head
        q = self.q_linear(x).reshape(B, -1, self.heads,
                                     C // self.heads).permute(0, 2, 1, 3)
        k = self.k_linear(x).reshape(B, -1, self.heads,
                                     C // self.heads).permute(0, 2, 1, 3)
        v = self.v_linear(x).reshape(B, -1, self.heads,
                                     C // self.heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x