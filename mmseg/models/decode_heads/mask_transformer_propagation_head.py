from genericpath import exists
import random
import math
import os
from importlib_metadata import requires
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import trunc_normal_

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from ..losses.accuracy import accuracy
from mmseg.ops import resize
from mmcv.runner import force_fp32

from timm.models.layers import DropPath
from mmcv.runner import get_dist_info


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
        downsample_rate=8,
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
        self.downsample_rate = downsample_rate

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, dropout, dpr[i])
            for i in range(n_layers)
        ])

        self.cls_emb_from_backbone = cls_emb_from_backbone
        if not cls_emb_from_backbone:
            # rank, _ = get_dist_info()
            # if rank == 0:
            #     self.cls_emb = torch.load(cls_emb_path, map_location="cpu")
            # # torch.cuda.synchronize()
            # if rank != 0: 
            #     self.cls_emb = torch.load(cls_emb_path, map_location="cpu")
            # # torch.cuda.synchronize()
            # self.cls_emb = self.cls_emb[: self.n_cls].reshape(
            #     1, self.n_cls, self.d_model).to(torch.device(f"cuda:{rank}"))
            self.cls_emb = torch.load(cls_emb_path, map_location="cpu")
            self.cls_emb.requires_grad = False

        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale *
                                       torch.randn(d_model, d_model))
        # self.proj_classes = nn.Parameter(self.scale *
        #                                  torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

    def init_weights(self):
        self.apply(init_weights)
        if not self.cls_emb_from_backbone:
            trunc_normal_(self.cls_emb, std=0.02)

    def forward(self, x):
        if self.cls_emb_from_backbone:
            x, cls_emb = x
        else:
            cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = self._transform_inputs(x)
        cls_emb = cls_emb.to(x.device)
        B, C, H, W = x.size()
        x = x.view(B, C, -1).permute(0, 2, 1)

        x = self.proj_dec(x)
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
        # logits = masks.clone()
        masks = self.mask_norm(masks)
        B, HW, N = masks.size()

        masks = masks.view(B, H, W, N).permute(0, 3, 1, 2)
        patches = patches.view(B, H, W, C).permute(0, 3, 1, 2)
        return masks, patches

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        masks, patches = self.forward(inputs)
        losses = self.losses(masks, patches, gt_semantic_seg)
        return losses

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_feat, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        h = seg_label.shape[-2] // self.downsample_rate
        w = seg_label.shape[-1] // self.downsample_rate
        seg_logit = resize(
            input=seg_logit,
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

        B, N, H, W = seg_logit.shape
        pos_bucket, prior_buckets = self._sample(seg_logit, seg_label)
        seg_weight = None
        seg_label = seg_label.squeeze(1)
        seg_logit = seg_logit.permute(0, 2, 3, 1).reshape(B * H * W, N)
        seg_label = seg_label.reshape(B * H * W)
        if len(prior_buckets) == 0:
            loss['loss_prior'] = torch.tensor(
                0, dtype=seg_logit.dtype, device=seg_logit.device, requires_grad=True
            )
            loss['loss_mask'] = torch.tensor(
                0, dtype=seg_logit.dtype, device=seg_logit.device, requires_grad=True
            )
        else:
            prior_inds = torch.cat(prior_buckets)
            loss['loss_prior'] = self.loss_decode(
                seg_logit[prior_inds],
                seg_label[prior_inds],
                weight=seg_weight,
                ignore_index=self.ignore_index)
            loss['loss_mask'] = self.propagation_loss(
                seg_feat, pos_bucket, prior_buckets
            )
            # print(loss['loss_prior'], loss['loss_mask'])
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss

    def propagation_loss(
        self, seg_feat, pos_bucket, prior_buckets, 
        sample_num=500, loss_weight=1
    ):
        """
        Args:
            seg_logit (torch.Tensor): segmentation logits, shape (B * H * W, N)
            seg_feat (torch.Tensor): segmentation feature, shape (B, C, H, W)
            cls_feat (torch.Tensor): (B, N, C)
        """
        B, C, H, W = seg_feat.shape
        seg_feat = seg_feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
        seg_feat = seg_feat / seg_feat.norm(dim=-1, keepdim=True)
        similarity = torch.tensor(
            0, dtype=seg_feat.dtype, device=seg_feat.device, requires_grad=True
        )
        valid_num = 0
        for pos_inds, prior_inds in zip(pos_bucket, prior_buckets):
            prior_inds = prior_inds.tolist()
            pos_inds = list(set(pos_inds.tolist()) - set(prior_inds))
            # pos_inds = random.sample(pos_inds, min(sample_num, len(pos_inds)))
            if len(pos_inds) == 0:
                continue
            cos_sim = seg_feat[prior_inds] @ seg_feat[pos_inds].transpose(0, 1)
            similarity = similarity + cos_sim.mean()
            
            valid_num += 1
        return 1 - (similarity / max(valid_num, 1))

    def _sample(self, seg_logit, seg_label, sample_rate=0.1, min_kept=10):
        """Sample pixels that have high loss or with low prediction confidence.

        Args:
            seg_logit (torch.Tensor): segmentation logits, shape (N, C, H, W)
            seg_label (torch.Tensor): segmentation label, shape (N, 1, H, W)

        Returns:
            torch.Tensor: segmentation weight, shape (N, H, W)
        """
        # with torch.no_grad():
        # gt_semantic_seg: B, 1, H, W
        # masks: B, N, H, W
        B, N, H, W = seg_logit.size()
        # assert B == 1, "Only support batch == 1 for segmenter!"
        seg_label = seg_label.reshape(B * H * W)
        # print(seg_logit.shape, seg_label.shape)
        unique_label = torch.unique(seg_label)
        unique_label = unique_label[unique_label != self.ignore_index]
        pos_bucket = []
        for l in unique_label:
            pos_bucket.append(
                (seg_label == l).nonzero(as_tuple=False)[:, 0]
            )
        
        # print("pos_bucket", pos_bucket)
        if len(pos_bucket) == 0:
            return [], []
        seg_logit = seg_logit.permute(0, 2, 3, 1).reshape(B * H * W, N)
        num_per_bucket = []
        for p in pos_bucket:
            k = int(sample_rate * len(p))
            if k < min_kept:
                k = min(min_kept, len(p))
            num_per_bucket.append(k)
        prior_bucket = []
        for k, p, l in zip(num_per_bucket, pos_bucket, unique_label):
            inds = seg_logit[p, int(l)].topk(k).indices
            # print("inds", inds)
            prior_bucket.append(
                p[inds]
            )
        # print("prior_bucket", prior_bucket)
        return pos_bucket, prior_bucket

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
