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
class MaskTransformerCosHead(BaseDecodeHead):
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
        upsample_input=1,
        downsample_rate=8,
        prior_rate=0.1,
        temperature=1.0,
        ann_suffix=".png",
        test_anno_dir="",
        use_pairwise_affinity=False,
        pairwise_affinity_thresh=0.95,
        cam_thresh=0.9,
        use_self_attention=False,
        reduce_zero_label=True,
        use_structure_loss=False,
        soft_structure_loss=False,
        structure_loss_weight=1.0,
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
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model**-0.5
        self.prior_loss_weight = 1.0
        self.downsample_rate = downsample_rate
        self.upsample_input = upsample_input
        self.prior_rate = prior_rate
        self.temperature = temperature
        self.ann_suffix = ann_suffix
        self.test_anno_dir = test_anno_dir
        # Pairwise Affinity for ImageNet21K supervision
        self.use_pairwise_affinity = use_pairwise_affinity
        self.pairwise_affinity_thresh = pairwise_affinity_thresh
        self.cam_thresh = cam_thresh
        self.use_self_attention = use_self_attention
        self.reduce_zero_label = reduce_zero_label
        self.use_structure_loss = use_structure_loss
        self.soft_structure_loss = soft_structure_loss
        self.structure_loss_weight = structure_loss_weight
        self.oracle_inference = oracle_inference
        self.num_oracle_points = num_oracle_points
        self.oracle_downsample_rate = oracle_downsample_rate

        self.proj_dec = nn.Linear(d_encoder, d_model)
        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.gamma = nn.Parameter(torch.ones([]))
        self.beta = nn.Parameter(torch.zeros([]))
        # NOTE: cosine classifier
        self.cls_emb = nn.Parameter(self.scale * torch.randn(self.num_classes, d_model))
        self.label_cos_sim = None

    def init_weights(self):
        self.apply(init_weights)

    def forward(self, x, img_metas, img_labels=None):
        x = self._transform_inputs(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        # NOTE: upsampling for RN101
        if self.upsample_input > 1:
            x = F.interpolate(
                x,
                scale_factor=self.upsample_input,
                mode="bilinear",
                align_corners=self.align_corners,
            )
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

        masks = patches @ cls_seg_feat.transpose(1, 2)
        if self.training:
            masks = (
                (masks - torch.mean(masks, dim=-1, keepdim=True))
                / torch.sqrt(
                    torch.var(masks, dim=-1, keepdim=True, unbiased=False) + 1e-5
                )
            ) * self.gamma + self.beta
        B, HW, N = masks.size()

        masks = masks.view(B, H, W, N).permute(0, 3, 1, 2)
        embeds = patches.clone()
        embeds = embeds.view(B, H, W, -1).permute(0, 3, 1, 2)
        return masks, embeds, feats

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        img_labels = None
        masks, embeds, feats = self.forward(inputs, img_metas)
        # assert False, f"{type(gt_semantic_seg), gt_semantic_seg.unique()}"
        losses = self.losses(masks, embeds, feats, gt_semantic_seg, img_labels)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg=None):
        masks, embeds, feats = self.forward(inputs, img_metas)
        if self.oracle_inference:
            # self.ignore_index = ignore_index
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
            mode="bilinear",
            align_corners=self.align_corners,
        )
        assert self.num_classes == 150
        seg_label = resize(input=seg_label.float(), size=(h, w), mode="nearest").long()[
            0, 0
        ]
        seg_label = seg_label - 1
        seg_label[seg_label == -1] = 255  # NOTE: hard code for convenience (ADE-150)
        seg_embed = seg_embed.permute(0, 2, 3, 1)
        seg_label_per_image = seg_label.reshape(h * w)
        seg_embed_per_image = seg_embed.reshape(h * w, C)
        seg_embed_per_image = seg_embed_per_image / seg_embed_per_image.norm(
            dim=-1, keepdim=True
        )
        unique_label = torch.unique(seg_label_per_image)
        unique_label = unique_label[unique_label != 255]
        masks = torch.zeros((B, self.num_classes, h, w), device=device)
        for l in unique_label:
            pos_inds = (seg_label_per_image == l).nonzero(as_tuple=False)[:, 0]
            inds = torch.randperm(len(pos_inds))[: self.num_oracle_points]
            prior_inds = pos_inds[inds]
            cos_mat = seg_embed_per_image[prior_inds] @ seg_embed_per_image.T
            score_mat = cos_mat.max(dim=0).values.reshape(h, w)
            masks[0, l] = score_mat
        return masks

    @force_fp32(apply_to=("seg_mask",))
    def losses(self, seg_mask, seg_embed, seg_feat, seg_label, img_labels=None):
        """Compute segmentation loss."""
        loss = dict()
        h = seg_label.shape[-2] // self.downsample_rate
        w = seg_label.shape[-1] // self.downsample_rate
        seg_mask = resize(
            input=seg_mask,
            size=(h, w),
            mode="bilinear",
            align_corners=self.align_corners,
        )
        seg_embed = resize(
            input=seg_embed,
            size=(h, w),
            mode="bilinear",
            align_corners=self.align_corners,
        )
        seg_feat = resize(
            input=seg_feat,
            size=(h, w),
            mode="bilinear",
            align_corners=self.align_corners,
        )
        seg_label = resize(input=seg_label.float(), size=(h, w), mode="nearest").long()

        B, N, H, W = seg_mask.shape
        prior_bucket = self._sample(seg_mask, seg_label)

        seg_label = seg_label.reshape(B * H * W)
        seg_mask = seg_mask.permute(0, 2, 3, 1).reshape(B * H * W, N)

        if len(prior_bucket) == 0:
            loss["loss_prior"] = torch.tensor(
                0, dtype=seg_mask.dtype, device=seg_mask.device, requires_grad=True
            )
            if self.use_structure_loss:
                loss["loss_structure"] = torch.tensor(
                    0, dtype=seg_mask.dtype, device=seg_mask.device, requires_grad=True
                )
        else:
            prior_inds = torch.cat(prior_bucket)
            loss["loss_prior"] = (
                self.loss_decode(
                    seg_mask[prior_inds],
                    seg_label[prior_inds],
                    weight=None,
                    ignore_index=self.ignore_index,
                )
                * self.prior_loss_weight
            )
            if self.use_structure_loss:
                loss["loss_structure"] = (
                    self.loss_structure(seg_feat, seg_label)
                    * self.structure_loss_weight
                )

        acc_weight = 1.0
        loss["acc_seg"] = accuracy(seg_mask, seg_label) * acc_weight
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
            pos_bucket.append((seg_label == l).nonzero(as_tuple=False)[:, 0])
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

    def loss_structure(self, seg_feat, seg_label):
        if self.soft_structure_loss and self.label_cos_sim is None:
            cls_emb = self.cls_emb.to(seg_feat.device).detach().clone()
            cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
            self.label_cos_sim = cls_emb @ cls_emb.T
            del cls_emb
        B, C, H, W = seg_feat.size()
        # seg_label = F.interpolate(seg_label.float(), size=(H, W)).long()
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
        sample_cls = torch.cat([seg_label[[i]] for i in pos_inds], dim=0).to(
            seg_feat.device
        )
        sample_feat = torch.cat([seg_feat[i] for i in pos_inds], dim=0)
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
        if self.soft_structure_loss:
            label_i = label.repeat_interleave(len(label)).long()
            label_j = label.repeat(len(label)).long()
            label_sim = self.label_cos_sim[label_i, label_j]
            label_sim = label_sim.reshape(len(label), len(label))
        valid_mask = torch.ones_like(cos_sim)
        valid_mask[
            torch.arange(len(valid_mask)).to(valid_mask.device),
            torch.arange(len(valid_mask)).to(valid_mask.device),
        ] = 0
        cos_sim = cos_sim[valid_mask.bool()]
        label_sim = label_sim[valid_mask.bool()]
        return torch.pow(cos_sim - label_sim, 2)

    def _sample_feat(self, buckets, total_sample_num=512):
        """Sample points from each buckets
        Args:
            num_per_buckets (list): number of points in each class
        """
        num_per_buckets = [len(p) for p in buckets]
        sample_per_bucket = [
            total_sample_num // len(buckets) for _ in range(len(num_per_buckets))
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
                target[i].data.float(), bins=nclass, min=0, max=nclass - 1
            )
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
        q = (
            self.q_linear(q)
            .reshape(B, -1, self.heads, C // self.heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_linear(k)
            .reshape(B, -1, self.heads, C // self.heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_linear(v)
            .reshape(B, -1, self.heads, C // self.heads)
            .permute(0, 2, 1, 3)
        )

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
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, cls_emb, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), cls_emb, cls_emb, mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention2(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim**-0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block2(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention2(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
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
