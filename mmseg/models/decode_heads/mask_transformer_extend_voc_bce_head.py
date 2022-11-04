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
        use_sample_class=False,
        num_smaple_class=100,
        basic_loss_weights=[0.2, 1.0],
        coseg_loss_weights=[0.2, 0.0], # for weak supervision
        use_pseudo_label=False,
        use_coseg=False,
        use_coseg_score_head=False,
        memory_bank_size=80,
        memory_bank_warm_up=100,
        foreground_topk=40,
        background_suppression=False,
        background_topk=5,
        background_thresh=0.2,
        background_mse_thresh=1.0, # MSE score
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
        self.use_sample_class = use_sample_class
        self.num_smaple_class = num_smaple_class
        self.basic_loss_weights = basic_loss_weights
        self.coseg_loss_weights = coseg_loss_weights
        self.weakly_supervised_datasets = weakly_supervised_datasets
        self.weakly_supervised = False
        self.use_pseudo_label = use_pseudo_label
        self.use_coseg = use_coseg
        self.use_coseg_score_head = use_coseg_score_head
        self.memory_bank_size = memory_bank_size
        self.memory_bank_warm_up = memory_bank_warm_up
        self.foreground_topk = foreground_topk
        self.background_suppression = background_suppression
        self.background_topk = background_topk
        self.background_thresh = background_thresh
        self.background_mse_thresh = background_mse_thresh
        # model parameters
        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.gamma = nn.Parameter(torch.ones([]))
        self.beta = nn.Parameter(torch.zeros([]))
        self.all_classes = self.num_classes if self.all_cls is None else len(self.all_cls)
        self.cls_emb = nn.Parameter(torch.randn(self.all_classes, d_model))
        if self.use_coseg:
            self.register_buffer(f"queue", torch.randn(self.all_classes, memory_bank_size, foreground_topk, d_model))
            self.register_buffer(f"ptr", torch.zeros(self.all_classes, dtype=torch.long))
            self.register_buffer(f"full", torch.zeros(self.all_classes, dtype=torch.long))
            self.queue = self.queue / self.queue.norm(dim=-1, keepdim=True)
        self.dim = 2 if self.background_suppression else 1
        # if self.use_coseg_score_head:
        #     self.coseg_head = FeedForward(dim=self.dim, hidden_dim=32)
        # else:
        self.coseg_head = normalize
        self.rank, self.world_size = get_dist_info()

    def init_weights(self):
        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    def _update(self, training, label=None):
        rank, _ = get_dist_info()
        if training:
            self.dataset_on_gpu = self.mix_batch_datasets[rank % len(self.mix_batch_datasets)]
            self.ignore_index = self.ignore_indices[rank % len(self.mix_batch_datasets)]
            self.basic_loss_weight = self.basic_loss_weights[rank % len(self.mix_batch_datasets)]
            self.coseg_loss_weight = self.coseg_loss_weights[rank % len(self.mix_batch_datasets)]
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
        elif self.dataset_on_gpu == "in11k":
            from mmseg.datasets.imagenet import ImageNet11K
            cls_name = ImageNet11K.CLASSES
        else:
            raise NotImplementedError(f"{self.dataset_on_gpu} is not supported")

        self.cls_name = cls_name
        if self.all_cls is not None:
            self.cls_index = [self.all_cls.index(name) for name in cls_name]
        else:
            self.cls_index = list(range(len(cls_name)))

        if training and self.use_sample_class:
            assert label is not None
            unique_label = torch.unique(label.flatten())
            unique_label = unique_label[unique_label != self.ignore_index].tolist()
            rand_inds = np.random.choice(len(self.cls_index), size=self.num_smaple_class, replace=False).tolist()
            rand_inds = list(set(rand_inds) | set(unique_label))
            self.cls_index = [self.cls_index[i] for i in rand_inds]
            remap_label = torch.zeros_like(label) + self.ignore_index
            for new_ind, old_ind in enumerate(rand_inds):
                if old_ind in unique_label:
                    remap_label[label==old_ind] = new_ind
            return remap_label
        else:
            return label

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
        gt_semantic_seg = self._update(training=True, label=gt_semantic_seg)
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
            loss = self.weakly_loss(seg_mask, seg_embed, seg_score, seg_label)
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
        loss["loss_coseg"] = seg_mask.sum() * 0.0
        return loss

    def weakly_loss(self, seg_mask, seg_embed, seg_score, seg_label):
        loss = dict()
        B, N, H, W = seg_mask.size()
        B, D, h, w = seg_embed.size()
        seg_mask = seg_mask.permute(0, 2, 3, 1).reshape(B, H * W, N)
        seg_embed = seg_embed.permute(0, 2, 3, 1).reshape(B, h * w, D)
        seg_score = seg_score.permute(0, 2, 3, 1).reshape(B, h * w, N)
        seg_label = seg_label.reshape(B, H * W)
        basic_loss, num_basic = 0.0, 0
        coseg_loss, num_coseg = 0.0, 0
        for mask, embed, score, label in zip(seg_mask, seg_embed, seg_score, seg_label):
            valid = label != self.ignore_index
            label_one_hot = F.one_hot(label[valid], num_classes=N).float().sum(dim=0)
            label_one_hot = (label_one_hot > 0).int().float()
            pred = mask[valid].mean(dim=0)
            if float(label_one_hot.sum()) > 0:
                basic_loss += F.binary_cross_entropy_with_logits(
                    pred, label_one_hot
                ) * self.basic_loss_weight
                num_basic += 1
                if self.use_coseg or self.use_pseudo_label:
                    coseg_loss += self._coseg_loss(
                        mask, score, embed, label, (h, w, H, W, N)
                    ) * self.coseg_loss_weight
                    num_coseg += 1
        if num_basic == 0:
            loss["loss_basic"] = seg_mask.sum() * 0.0 
        else:
            loss["loss_basic"] = basic_loss / num_basic
        if num_coseg == 0:
            loss["loss_coseg"] = seg_mask.sum() * 0.0 
        else:
            loss["loss_coseg"] = coseg_loss / num_coseg
        return loss

    def _coseg_loss(self, mask, score, embed, label, shape):
        h, w, H, W, N = shape
        # (h * w, 1) or (h * w, 2)
        unique_label = torch.unique(label).flatten()
        unique_label = unique_label[unique_label != self.ignore_index].tolist()
        coseg_score_mat = torch.ones_like(mask)
        for fg_label in unique_label:
            if self.use_coseg:
                bg_labels = [x for x in unique_label if x != fg_label]
                coseg_score = self._coseg_score(score, embed, fg_label, bg_labels)
                try:
                    self._dequeue_and_enqueue(embed, score, fg_label)
                except:
                    print("Unsuccessful _dequeue_and_enqueue!")
                if coseg_score is None: continue
                if self.background_suppression and not self.use_coseg_score_head:
                    coseg_score = coseg_score[...,0] - coseg_score[...,1]
                coseg_score = self.coseg_head(coseg_score).reshape(h, w)
                coseg_score = F.interpolate(
                    coseg_score[None, None], size=(H, W), mode="bilinear", align_corners=self.align_corners
                )[0, 0].flatten()
                coseg_score_mat[:, fg_label] = coseg_score.sigmoid()
            elif self.use_pseudo_label:
                score_fg = score[:, fg_label]
                thresh = score_fg.mean() + score_fg.std()
                bg_inds = (score_fg < thresh).nonzero(as_tuple=False).flatten()
                coseg_score_mat[bg_inds, fg_label] = 0.0
        mask = mask * coseg_score_mat
        # calculate bce loss
        valid = label != self.ignore_index
        label_one_hot = F.one_hot(label[valid], num_classes=N).float().sum(dim=0)
        label_one_hot = (label_one_hot > 0).int().float()
        pred = mask[valid].mean(dim=0)
        return F.binary_cross_entropy_with_logits(
            pred, label_one_hot
        )

    def _coseg_score(self, score, embed, fg_label, bg_labels):
        fg_ind = self.cls_index[fg_label]
        if int(self.full[fg_ind]) < self.memory_bank_warm_up:
            return None
        fg_embed = self.queue[fg_ind].reshape(-1, self.d_model)
        fg_embed = fg_embed / fg_embed.norm(dim=-1, keepdim=True)
        embed = embed / embed.norm(dim=-1, keepdim=True)
        cos_mat = embed @ fg_embed.T
        coseg_score = cos_mat.mean(dim=-1)
        if not self.background_suppression:
            return coseg_score[:, None]
        # bg_scores = score.topk(self.background_topk, dim=0).values.mean(dim=0)
        if len(bg_labels) > 0:
            bg_classes = bg_labels
        else:
            bg_classes = []
            # bg_classes = (bg_scores > self.background_thresh).nonzero(as_tuple=False)
            # bg_classes = bg_classes.flatten().tolist()
            # if fg_label in bg_classes:
            #     bg_classes.remove(fg_label)
        bg_inds = [self.cls_index[bg_cls] for bg_cls in bg_classes]
        if len(bg_inds) > 0:
            bg_embed = self.queue[bg_inds]
            bg_embed = bg_embed.reshape(-1, self.d_model)
            bg_embed = bg_embed / bg_embed.norm(dim=-1, keepdim=True)
            cos_mat = embed @ bg_embed.T
            bg_coseg_score = cos_mat.reshape(len(embed), len(bg_inds), -1).mean(dim=-1)
        if len(bg_inds) > 0 and len(bg_labels) == 0 and bg_coseg_score.numel() > 0:
            mse_score = mse(coseg_score[:, None], bg_coseg_score)
            bg_coseg_score = bg_coseg_score[:, mse_score > self.background_mse_thresh]
        if len(bg_inds) > 0 and bg_coseg_score.numel() > 0:
            bg_coseg_score = bg_coseg_score.mean(dim=-1)
            coseg_score = torch.stack([coseg_score, bg_coseg_score], dim=-1)
        if coseg_score.dim() == 1:
            coseg_score = torch.stack([coseg_score, 1.0 - coseg_score], dim=-1)
        return coseg_score

    def _dequeue_and_enqueue(self, embed, score, cls):
        """
        Args:
            embed: torch.Tensor(h * w, D)
            score: torch.Tensor(h * w, N)
            cls: int, class index of current dataset
        """
        inds = score[:, cls].topk(self.foreground_topk).indices.flatten()
        embed = embed[inds]
        cls_ind = self.cls_index[cls]
        ptr = int(self.ptr[cls_ind])
        self.queue[cls_ind, ptr] = embed
        if (ptr + 1) >= self.memory_bank_size:
            self.full[cls_ind] += 1
        ptr = (ptr + 1) % self.memory_bank_size
        self.ptr[cls_ind] = ptr

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
