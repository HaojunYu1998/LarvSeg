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
from ..builder import HEADS, build_loss
from ..losses.accuracy import accuracy
from .decode_head import BaseDecodeHead


def mse(img1, img2):
    img1 = (img1 - img1.mean(dim=0, keepdim=True)) / img1.std(dim=0, keepdim=True)
    img2 = (img2 - img2.mean(dim=0, keepdim=True)) / img2.std(dim=0, keepdim=True)
    return torch.pow(img1 - img2, 2).mean(dim=0)


def normalize(x):
    return (x - torch.mean(x, dim=-1, keepdim=True)) / torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5)


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


@HEADS.register_module()
class MaskTransformerExtendVocBCEHead(BaseDecodeHead):

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
        all_cls_path="",
        mix_batch_datasets=["in124", "coco171"],
        weakly_supervised_datasets=["in124"],
        test_dataset="ade124",
        ignore_indices=[255, 255],
        test_ignore_index=255,
        basic_loss_weights=[0.2, 1.0],
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
        assert len(mix_batch_datasets) == len(basic_loss_weights)
        self.d_encoder = d_encoder
        self.n_cls = n_cls
        self.d_model = d_model
        self.scale = d_model**-0.5
        self.downsample_rate = downsample_rate
        self.all_cls = None
        if os.path.exists(all_cls_path):
            self.all_cls = json.load(open(all_cls_path))
        self.mix_batch_datasets = mix_batch_datasets
        self.test_dataset = test_dataset
        self.ignore_indices = ignore_indices
        self.test_ignore_index = test_ignore_index
        self.basic_loss_weights = basic_loss_weights
        self.weakly_supervised_datasets = weakly_supervised_datasets
        self.weakly_supervised = False
        # model parameters
        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.gamma = nn.Parameter(torch.ones([]))
        self.beta = nn.Parameter(torch.zeros([]))
        self.all_classes = self.num_classes if self.all_cls is None else len(self.all_cls)
        self.cls_emb = nn.Parameter(torch.randn(self.all_classes, d_model))
        self.rank, self.world_size = get_dist_info()

    def init_weights(self):
        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    def _update(self, training):
        rank, _ = get_dist_info()
        if training:
            self.dataset_on_gpu = self.mix_batch_datasets[rank % len(self.mix_batch_datasets)]
            self.ignore_index = self.ignore_indices[rank % len(self.mix_batch_datasets)]
            self.basic_loss_weight = self.basic_loss_weights[rank % len(self.mix_batch_datasets)]
            self.weakly_supervised = self.dataset_on_gpu in self.weakly_supervised_datasets
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
        elif self.dataset_on_gpu == "ade124":
            from mmseg.datasets.ade import ADE20K124Dataset
            cls_name = ADE20K124Dataset.CLASSES124
        elif self.dataset_on_gpu == "ade130":
            from mmseg.datasets.ade import ADE20K130Dataset
            cls_name = ADE20K130Dataset.CLASSES130
        elif self.dataset_on_gpu == "ade847":
            from mmseg.datasets.ade import ADE20KFULLDataset
            cls_name = ADE20KFULLDataset.CLASSES
        elif self.dataset_on_gpu == "ade585":
            from mmseg.datasets.ade import ADE20K585Dataset
            cls_name = ADE20K585Dataset.CLASSES585
        elif self.dataset_on_gpu == "in124":
            from mmseg.datasets.imagenet import ImageNet124
            cls_name = ImageNet124.CLASSES
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

    def _mask_norm(self, masks):
        return (
            (masks - torch.mean(masks, dim=-1, keepdim=True))
            / torch.sqrt(torch.var(masks, dim=-1, keepdim=True, unbiased=False) + 1e-5)
        ) * self.gamma + self.beta

    def forward(self, x, img_metas):
        x = self._transform_inputs(x)
        B, D, H, W = x.size()
        x = x.view(B, D, -1).permute(0, 2, 1)
        cls_emb = self.cls_emb[self.cls_index]
        cls_emb = cls_emb.expand(x.size(0), -1, -1)
        patches = x
        patches = patches @ self.proj_patch
        cls_emb = cls_emb @ self.proj_classes
        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
        masks = patches @ cls_emb.transpose(1, 2)
        scores = masks.clone().detach()
        embeds = patches.clone().detach()
        masks = self._mask_norm(masks)
        B, HW, N = masks.size()
        masks = masks.view(B, H, W, N).permute(0, 3, 1, 2)
        scores = scores.view(B, H, W, N).permute(0, 3, 1, 2)
        embeds = embeds.view(B, H, W, -1).permute(0, 3, 1, 2)
        return masks, embeds, scores

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        self._update(training=True)
        masks, embeds, scores = self.forward(inputs, img_metas)
        losses = self.losses(masks, embeds, scores, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg=None, img=None):
        self._update(training=False)
        masks, _, _ = self.forward(inputs, img_metas)
        return masks

    @force_fp32(apply_to=('seg_mask', ))
    def losses(self, seg_mask, seg_embed, seg_score, seg_label):
        """Compute segmentation loss."""
        h = seg_label.shape[-2] // self.downsample_rate
        w = seg_label.shape[-1] // self.downsample_rate
        seg_mask = resize(
            seg_mask, size=(h, w), mode='bilinear', align_corners=self.align_corners
        )
        seg_label = resize(
            seg_label.float(), size=(h, w), mode='nearest'
        ).long()
        if self.weakly_supervised:
            loss = self.weakly_loss(seg_mask, seg_label)
        else:
            loss = self.supervised_loss(seg_mask, seg_label)
        loss['acc_seg'] = self._log_accuracy(seg_mask, seg_label)
        return loss
    
    def _log_accuracy(self, seg_mask, seg_label):
        B, N, H, W = seg_mask.shape
        seg_label = seg_label.flatten()
        seg_mask = seg_mask.permute(0, 2, 3, 1).reshape(B * H * W, N)
        weak_in_batch = len(self.weakly_supervised_datasets) > 0
        acc_weight = 1.0
        if weak_in_batch:
            num_datasets = len(self.mix_batch_datasets)
            all_data = [self.mix_batch_datasets[r % num_datasets] for r in range(self.world_size)]
            sup_data = [d for d in all_data if d not in self.weakly_supervised_datasets]
            acc_mult_weight = len(all_data) / len(sup_data)
            acc_weight = 0.0 if self.weakly_supervised else acc_mult_weight
        return accuracy(seg_mask, seg_label) * acc_weight
    
    # def supervised_loss(self, seg_mask, seg_label):
    #     """
    #     Args:
    #         seg_mask (torch.Tensor): segmentation logits, shape (B, N, H, W)
    #         seg_label (torch.Tensor): segmentation label, shape (B, 1, H, W)
    #     """
    #     loss = dict()
    #     B, N, H, W = seg_mask.size()
    #     seg_mask = seg_mask.permute(0, 2, 3, 1).reshape(B * H * W, N)
    #     seg_label = seg_label.reshape(B * H * W)
    #     valid = seg_label != self.ignore_index
    #     seg_mask = seg_mask[valid]
    #     seg_label = seg_label[valid]
    #     seg_label = F.one_hot(seg_label, num_classes=N).float()
    #     loss["loss_basic"] = F.binary_cross_entropy_with_logits(
    #         seg_mask, seg_label
    #     ) * self.basic_loss_weight
    #     return loss

    def supervised_loss(self, seg_mask, seg_label):
        """
        Args:
            seg_mask (torch.Tensor): segmentation logits, shape (B, N, H, W)
            seg_label (torch.Tensor): segmentation label, shape (B, 1, H, W)
        """
        loss = dict()
        B, N, H, W = seg_mask.size()
        seg_mask = seg_mask.permute(0, 2, 3, 1).reshape(B * H * W, N)
        seg_label = seg_label.reshape(B * H * W)
        loss["loss_basic"] = self.loss_decode(
            seg_mask, seg_label, ignore_index=self.ignore_index
        ) * self.basic_loss_weight
        return loss

    def weakly_loss(self, seg_mask, seg_label):
        loss = dict()
        B, N, H, W = seg_mask.size()
        seg_mask = seg_mask.permute(0, 2, 3, 1).reshape(B, H * W, N)
        seg_label = seg_label.reshape(B, H * W)
        basic_loss, num_basic = 0.0, 0
        for mask, label in zip(seg_mask, seg_label):
            valid = label != self.ignore_index
            label_one_hot = F.one_hot(label[valid], num_classes=N).float().sum(dim=0)
            label_one_hot = (label_one_hot > 0).int().float()
            pred = mask[valid].mean(dim=0)
            if float(label_one_hot.sum()) > 0:
                basic_loss += F.binary_cross_entropy_with_logits(
                    pred, label_one_hot
                ) * self.basic_loss_weight
                num_basic += 1
        if num_basic == 0:
            loss["loss_basic"] = seg_mask.sum() * 0.0 
        else:
            loss["loss_basic"] = basic_loss / num_basic
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
