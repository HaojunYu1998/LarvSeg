from genericpath import exists
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import trunc_normal_

from ..builder import HEADS
from .decode_head import BaseDecodeHead
from timm.models.layers import DropPath


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


@HEADS.register_module()
class MaskTransformerPixEmbedHead(BaseDecodeHead):

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
        pixemb_before_attn=False,
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
        self.pixemb_before_attn = pixemb_before_attn

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, dropout, dpr[i])
            for i in range(n_layers)
        ])

        self.cls_emb_from_backbone = cls_emb_from_backbone
        if not cls_emb_from_backbone:
            self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale *
                                       torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale *
                                         torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        if self.pixemb_before_attn:
            self.pixemb_blocks = nn.ModuleList([
                Block(d_model, n_heads, d_ff, dropout, dpr[i])
                for i in range(n_layers)
            ])
            self.pixemb_norm = nn.LayerNorm(d_model)

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
        B, C, H, W = x.size()
        x = x.view(B, C, -1).permute(0, 2, 1)

        if self.pixemb_before_attn:
            embeds = x.clone() # (B, H * W, C)
            for blk in self.pixemb_blocks:
                embeds = blk(embeds)
            embeds = self.pixemb_norm(embeds)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)

        x = self.proj_dec(x)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, :-self.n_cls], x[:, -self.n_cls:]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        # B, HW, C
        patches = patches / patches.norm(dim=-1, keepdim=True)
        # B, N, C
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        logits = masks.clone()
        masks = self.mask_norm(masks)
        B, HW, N = masks.size()

        masks = masks.view(B, H, W, N).permute(0, 3, 1, 2)
        # patches = patches.view(B, H, W, C).permute(0, 3, 1, 2)
        if not self.pixemb_before_attn:
            embeds = patches.clone()
        embeds = embeds.view(B, H, W, C).permute(0, 3, 1, 2)
        return masks, logits, embeds

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        masks, logits, embeds = self.forward(inputs)
        losses = self.losses(masks, gt_semantic_seg)
        losses.update(
            self.pix_embed_losses(embeds, gt_semantic_seg)
        )
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        masks, logits, embeds = self.forward(inputs)
        thresh_file = test_cfg.get("logit_thresh", None)
        if thresh_file is not None:
            device = masks.device
            thresh = torch.load(thresh_file).to(device)[:, None, None]
            mask_list = []
            for mask, embed in zip(masks, embeds):
                mask = self.graph_cut_inference(mask, embed, thresh)
                mask_list.append(mask) # N, H, W
            masks = torch.stack(mask_list, dim=0).to(device)
        # save feature map
        save_feature_dir = test_cfg.get("save_feature_dir", None)
        if save_feature_dir is not None:
            save_feature_dir = os.path.join(os.getcwd(), save_feature_dir)
            os.makedirs(save_feature_dir, exist_ok=True)
            ori_filename = img_metas[0]["ori_filename"]
            save_path = os.path.join(
                save_feature_dir, ori_filename.replace(".jpg", ".pth")
            )
            torch.save(
                embeds.detach().cpu().half(), save_path
            )
        del embeds
        # save logits
        save_logit_dir = test_cfg.get("save_logit_dir", None)
        if save_logit_dir is not None:
            save_logit_dir = os.path.join(os.getcwd(), save_logit_dir)
            os.makedirs(save_logit_dir, exist_ok=True)
            ori_filename = img_metas[0]["ori_filename"]
            save_path = os.path.join(
                save_logit_dir, ori_filename.replace(".jpg", ".pth")
            )
            torch.save(
                logits.detach().cpu().half(), save_path
            )
        del logits
        return masks

    def graph_cut_inference(self, logit, pix_embedding, thresh):
        """
        Params:
            logit: N, H, W
            pix_embedding: C, H, W
            thresh: N, 1, 1
        """
        N, H, W = logit.shape
        # thresh = logit.reshape(N, -1).topk(100, dim=1).values[:, -1][:, None, None]
        valid_masks = logit > thresh
        # max_mask = logit == logit.max(dim=0).values[None]
        # valid_masks = valid_masks & max_mask
        # logit[~valid_masks] = -float("inf")
        # ignore = torch.zeros_like(logit)[[0]] - 100.0
        # return torch.cat([logit, ignore], dim=0)
        # (H * W, C)
        pix_embedding = pix_embedding.reshape(-1, H * W).transpose(0, 1)
        cos_sims = []
        for valid_mask in valid_masks:
            if not valid_mask.any():
                continue
            cos_sim = pix_embedding[valid_mask.flatten()] @ pix_embedding.T
            cos_sim = cos_sim.max(dim=0).values
            cos_sims.append(cos_sim)
        # (N_valid, H * W) => (H, W)
        pred = torch.stack(cos_sims, dim=0).argmax(dim=0).reshape(H, W)
        full_perd = torch.zeros(N, H, W).to(logit.device)
        valid_cls = valid_masks.sum(dim=[1,2]).nonzero(as_tuple=False).flatten()
        for i, cls in enumerate(valid_cls):
            full_perd[cls] = (pred == i).float()
        return full_perd

    def pix_embed_losses(self, seg_feature, seg_label):
        """
        Params:
            seg_feature: B, C, H, W
            seg_label: B, 1, H_, W_
        """
        B, C, H, W = seg_feature.size()
        seg_label = F.interpolate(seg_label.float(), size=(H, W)).long()
        seg_feature = seg_feature.permute(0, 2, 3, 1).reshape(B * H * W, C)
        seg_label = seg_label.reshape(B * H * W)
        unique_label = torch.unique(seg_label)
        pos_bucket = [
            torch.nonzero(seg_label == l)[:, 0]
            for l in unique_label
            if l != self.ignore_index
        ]
        if len(pos_bucket) == 0:
            return {
                "loss_emb": seg_feature[seg_label != self.ignore_index].sum()
            }
        pos_inds = self._sample(pos_bucket)
        sample_cls = torch.cat(
            [torch.Tensor([i for _ in range(len(p))]) for i, p in enumerate(pos_inds)],
            dim=0,
        ).to(seg_feature.device)
        sample_embed = torch.cat([seg_feature[i] for i in pos_inds], dim=0)
        loss = self.loss_similarity(sample_embed, sample_cls)
        return {"loss_emb": loss} 

    def _sample(self, buckets, total_sample_num=512):
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

    def loss_similarity(self, embedding, label, temperature=0.02):
        """Compute the similarity loss
        Args:
            embedding (torch.Tensor): [B,C]
            label (torch.Tensor): [B]
        """
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        cos_sim = embedding @ embedding.T  # [B,B]
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
            self.qkv(x).reshape(B, N, 3, self.heads,
                                C // self.heads).permute(2, 0, 3, 1, 4))
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


class Block(nn.Module):

    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

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
