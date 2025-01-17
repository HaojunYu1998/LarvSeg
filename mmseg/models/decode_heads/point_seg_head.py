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


# TODO: Single-image Category-wise Attentive Classifier
# TODO: Only FG-Enhance Attentive Classifier
# TODO: Only BG-Suppress Attentive Classifier
# TODO: C171+I124 => eval A124, OpenVoc Baselines, simple baseline, LarvSeg
# TODO: C171+I585 => eval A585, OpenVoc Baselines, simple baseline, LarvSeg
# TODO: Swin or ResNet for backbone
# TODO: Attentive and Segmentation classifier share the same weights, simple baseline uses seperate weights
# TODO: Pascal Context 459, COCO-LVIS 1284


def mse(img1, img2):
    img1 = (img1 - img1.mean(dim=0, keepdim=True)) / img1.std(dim=0, keepdim=True)
    img2 = (img2 - img2.mean(dim=0, keepdim=True)) / img2.std(dim=0, keepdim=True)
    return torch.pow(img1 - img2, 2).mean(dim=0)


def normalize(x):
    return (x - torch.mean(x, dim=-1, keepdim=True)) / torch.sqrt(
        torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5
    )


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


@HEADS.register_module()
class PointSegHead(BaseDecodeHead):
    def __init__(
        self,
        n_cls,  # for evaluation
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
        ignore_cls_path="",
        mix_batch_datasets=["in124", "coco171"],
        weakly_supervised_datasets=["in124"],
        test_dataset="ade124",
        ignore_indices=[255, 255],
        test_ignore_index=255,
        use_lang_seg=False,
        cls_emb_train="notebook/cls_emb_c171.pth",
        cls_emb_test="notebook/cls_emb_a150.pth",
        use_sample_class=False,
        num_smaple_class=100,
        basic_loss_weights=[0.2, 1.0],
        coseg_loss_weights=[0.2, 0.0],  # for weak supervision
        use_coseg=False,
        use_coseg_single_image=False,
        use_coseg_inference=False,
        use_coseg_score_head=False,
        use_weak_basic_loss=True,
        use_weak_bce_loss=False,
        use_pseudo_label=False,
        memory_bank_size=80,
        memory_bank_warm_up=100,
        foreground_topk=40,
        background_suppression=False,
        background_topk=5,
        background_thresh=0.2,
        background_mse_thresh=1.0,  # MSE score
        oracle_inference=False,
        num_oracle_points=1,
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
        assert len(mix_batch_datasets) == len(basic_loss_weights)
        self.d_encoder = d_encoder
        self.n_cls = n_cls
        self.d_model = d_model
        self.scale = d_model**-0.5
        self.downsample_rate = downsample_rate
        self.all_cls = None
        self.ignore_cls = None
        if os.path.exists(all_cls_path):
            self.all_cls = json.load(open(all_cls_path))
        if os.path.exists(ignore_cls_path):
            assert len(weakly_supervised_datasets) == 0
            assert len(mix_batch_datasets) == 1
            self.ignore_cls = json.load(open(ignore_cls_path))
        self.mix_batch_datasets = mix_batch_datasets
        self.test_dataset = test_dataset
        self.ignore_indices = ignore_indices
        self.test_ignore_index = test_ignore_index
        self.use_lang_seg = use_lang_seg
        self.cls_emb_train = cls_emb_train
        self.cls_emb_test = cls_emb_test
        self.use_sample_class = use_sample_class
        self.num_smaple_class = num_smaple_class
        self.basic_loss_weights = basic_loss_weights
        self.coseg_loss_weights = coseg_loss_weights
        self.weakly_supervised_datasets = weakly_supervised_datasets
        self.weakly_supervised = False
        self.use_coseg = use_coseg
        self.use_coseg_single_image = use_coseg_single_image
        self.use_coseg_inference = use_coseg_inference
        self.use_coseg_score_head = use_coseg_score_head
        self.use_weak_basic_loss = use_weak_basic_loss
        self.use_weak_bce_loss = use_weak_bce_loss
        self.use_pseudo_label = use_pseudo_label
        self.memory_bank_size = memory_bank_size
        self.memory_bank_warm_up = memory_bank_warm_up
        self.foreground_topk = foreground_topk
        self.background_suppression = background_suppression
        self.background_topk = background_topk
        self.background_thresh = background_thresh
        self.background_mse_thresh = background_mse_thresh
        self.oracle_inference = oracle_inference
        self.num_oracle_points = num_oracle_points
        self.oracle_downsample_rate = oracle_downsample_rate
        # model parameters
        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.gamma = nn.Parameter(torch.ones([]))
        self.beta = nn.Parameter(torch.zeros([]))
        self.all_classes = (
            self.num_classes if self.all_cls is None else len(self.all_cls)
        )
        if not self.use_lang_seg:
            self.cls_emb = nn.Parameter(torch.randn(self.all_classes, d_model))
        self.dim = 2 if self.background_suppression else 1
        self.coseg_head = normalize
        self.rank, self.world_size = get_dist_info()

    def init_weights(self):
        self.apply(init_weights)
        if not self.use_lang_seg:
            trunc_normal_(self.cls_emb, std=0.02)

    def _update(self, training, label=None):
        rank, _ = get_dist_info()

        if training:
            self.dataset_on_gpu = self.mix_batch_datasets[
                rank % len(self.mix_batch_datasets)
            ]
            self.ignore_index = self.ignore_indices[rank % len(self.mix_batch_datasets)]
            self.basic_loss_weight = self.basic_loss_weights[
                rank % len(self.mix_batch_datasets)
            ]
            self.coseg_loss_weight = self.coseg_loss_weights[
                rank % len(self.mix_batch_datasets)
            ]
            self.weakly_supervised = (
                self.dataset_on_gpu in self.weakly_supervised_datasets
            )
        else:
            self.dataset_on_gpu = self.test_dataset
            self.ignore_index = self.test_ignore_index

        if self.use_lang_seg:
            cls_emb_path = self.cls_emb_train if training else self.cls_emb_test
            if isinstance(cls_emb_path, list):
                cls_emb_path = cls_emb_path[rank % len(self.mix_batch_datasets)]
            self.cls_emb = torch.load(cls_emb_path, map_location="cpu")
            self.cls_emb.requires_grad = False

        if self.dataset_on_gpu == "coco171":
            from mmseg.datasets.coco_stuff import COCOStuffDataset, ProcessedC171Dataset

            cls_name = COCOStuffDataset.CLASSES
            if len(self.mix_batch_datasets) > 1 and \
                "coco171" not in self.weakly_supervised_datasets:
                cls_name = ProcessedC171Dataset.CLASSES
                cls_name = [x.split("-")[0] for x in cls_name]
        elif self.dataset_on_gpu == "pc59":
            from mmseg.datasets.pascal_context import PascalContextDataset59

            cls_name = PascalContextDataset59.CLASSES
        elif self.dataset_on_gpu == "pc459":
            from mmseg.datasets.pascal_context import PascalContextDataset459

            cls_name = PascalContextDataset459.CLASSES
        elif self.dataset_on_gpu == "city19":
            from mmseg.datasets.cityscapes import CityscapesDataset

            cls_name = CityscapesDataset.CLASSES
        elif self.dataset_on_gpu == "ade150":
            from mmseg.datasets.ade import ADE20KDataset

            cls_name = ADE20KDataset.CLASSES
        elif self.dataset_on_gpu == "ade124":
            from mmseg.datasets.ade import ADE20K124Dataset

            cls_name = ADE20K124Dataset.CLASSES124
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

        if training and self.ignore_cls is not None:
            assert label is not None
            unique_label = torch.unique(label.flatten())
            unique_label = unique_label[unique_label != self.ignore_index].tolist()
            kept_inds = [
                i for i, c in enumerate(self.cls_name) if c not in self.ignore_cls
            ]
            map_dict = {old_ind: new_ind for new_ind, old_ind in enumerate(kept_inds)}
            remap_label = torch.zeros_like(label) + self.ignore_index
            for l in unique_label:
                remap_label[label == l] = map_dict[l]
            label = remap_label
            self.cls_index = [self.cls_index[i] for i in kept_inds]
            self.cls_name = [self.cls_name[i] for i in kept_inds]

        if training and self.use_sample_class:
            assert label is not None
            unique_label = torch.unique(label.flatten())
            unique_label = unique_label[unique_label != self.ignore_index].tolist()
            if len(self.cls_index) >self.num_smaple_class:
                rand_inds = np.random.choice(
                    len(self.cls_index), size=self.num_smaple_class, replace=False
                ).tolist()
                rand_inds = list(set(rand_inds) | set(unique_label))
                self.cls_index = [self.cls_index[i] for i in rand_inds]
                remap_label = torch.zeros_like(label) + self.ignore_index
                for new_ind, old_ind in enumerate(rand_inds):
                    if old_ind in unique_label:
                        remap_label[label == old_ind] = new_ind
            else:
                remap_label =label
            return remap_label
        else:
            return label

    def _mask_norm(self, masks):
        return (
            (masks - torch.mean(masks, dim=-1, keepdim=True))
            / torch.sqrt(torch.var(masks, dim=-1, keepdim=True, unbiased=False) + 1e-5)
        ) * self.gamma + self.beta

    def forward(self, x):
        x = self._transform_inputs(x)
        B, D, H, W = x.size()
        x = x.view(B, D, -1).permute(0, 2, 1)
        cls_emb = self.cls_emb[self.cls_index]
        cls_emb = cls_emb.expand(x.size(0), -1, -1)
        cls_emb = cls_emb.to(x.device)
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

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg=None):
        gt_semantic_seg = self._update(training=True, label=gt_semantic_seg)
        masks, embeds, scores = self.forward(inputs)
        losses = self.losses(masks, embeds, scores, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg=None, img=None):
        self._update(training=False)
        masks, embeds, scores = self.forward(inputs)
        return masks

    @force_fp32(apply_to=("seg_mask",))
    def losses(self, seg_mask, seg_embed, seg_score, seg_label):
        """Compute segmentation loss."""
        h = seg_label.shape[-2] // self.downsample_rate
        w = seg_label.shape[-1] // self.downsample_rate
        seg_mask = resize(
            seg_mask, size=(h, w), mode="bilinear", align_corners=self.align_corners
        )
        seg_label = resize(seg_label.float(), size=(h, w), mode="nearest").long()
        if self.weakly_supervised:
            loss = self.weakly_loss(seg_mask, seg_embed, seg_score, seg_label)
        else:
            loss = self.supervised_loss(seg_mask, seg_label)
        loss["acc_seg"] = self._log_accuracy(seg_mask, seg_label)
        return loss

    def _log_accuracy(self, seg_mask, seg_label):
        B, N, H, W = seg_mask.shape
        seg_label = seg_label.flatten()
        seg_mask = seg_mask.permute(0, 2, 3, 1).reshape(B * H * W, N)
        weak_in_batch = len(self.weakly_supervised_datasets) > 0
        acc_weight = 1.0
        if weak_in_batch:
            num_datasets = len(self.mix_batch_datasets)
            all_data = [
                self.mix_batch_datasets[r % num_datasets]
                for r in range(self.world_size)
            ]
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
        loss["loss_basic"] = (
            self.loss_decode(seg_mask, seg_label, ignore_index=self.ignore_index)
            * self.basic_loss_weight
        )
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
        
        for idx, (mask, embed, score, label) in enumerate(
            zip(seg_mask, seg_embed, seg_score, seg_label)
        ):
            unique_label = torch.unique(label)
            unique_label = unique_label[unique_label != self.ignore_index].tolist()

            # construct cos_mat_dict
            cos_mat_dict = {}
            with torch.no_grad():
                embed = embed / embed.norm(dim=-1, keepdim=True)
                embed = F.interpolate(
                    embed.reshape(1, h, w, D).permute(0, 3, 1, 2), size=(H, W)
                ).permute(0, 2, 3, 1).reshape(H * W, D)
                for l in unique_label:
                    embed_l = embed[label == l]
                    idx = np.random.choice(len(embed_l), 1)
                    embed_l = embed_l[[idx]]
                    embed_l = embed_l / embed_l.norm(dim=-1, keepdim=True)
                    cos_mat = embed @ embed_l.T
                    cos_mat_dict[l] = cos_mat.mean(dim=-1)

            if self.use_pseudo_label:
                coseg_loss += self._single_pseudo_label_loss(
                    mask, label, (h, w, H, W, N), cos_mat_dict
                )
                num_coseg += 1
                continue

            # bce_loss
            if self.use_weak_bce_loss and self.use_weak_basic_loss:
                valid = label != self.ignore_index
                label_one_hot = F.one_hot(label[valid], num_classes=N).float().sum(dim=0)
                label_one_hot = (label_one_hot > 0).int().float()
                pred = mask[valid].mean(dim=0)
                if float(label_one_hot.sum()) > 0:
                    basic_loss += (
                        F.binary_cross_entropy_with_logits(pred, label_one_hot)
                    )
                    num_basic += 1
            if self.use_weak_bce_loss:
                # mask, label, shape, cos_mat_dict
                coseg_loss += self._single_coseg_bce_loss(
                    mask, label, (h, w, H, W, N), cos_mat_dict
                )
                num_coseg += 1
                continue
            
            # ce_loss
            for l in unique_label:
                ignore_inds = [x for x in unique_label if x != l]
                if self.use_weak_basic_loss:
                    basic_loss += self._cross_entropy_loss(mask, l, ignore_inds)
                    num_basic += 1
                coseg_loss += self._single_coseg_loss(
                    mask, l, (h, w, H, W), ignore_inds, cos_mat_dict
                )
                num_coseg += 1
        if num_basic == 0:
            loss["loss_basic"] = seg_mask.sum() * 0.0
        else:
            assert self.use_weak_basic_loss
            loss["loss_basic"] = basic_loss / num_basic * self.basic_loss_weight
        if num_coseg == 0:
            loss["loss_coseg"] = seg_mask.sum() * 0.0
        else:
            loss["loss_coseg"] = coseg_loss / num_coseg * self.coseg_loss_weight
        return loss

    def _cross_entropy_loss(self, pred, label, ignore_indices):
        """Cross Entropy Loss with multiple ignore indices."""
        assert pred.numel() > 0, "no elements in prediction"
        if "in" in self.dataset_on_gpu:
            assert len(ignore_indices) == 0, f"more than one classes for imagenet data"
        N = pred.shape[-1]
        kept_indices = [i for i in range(N) if i not in ignore_indices]
        if len(kept_indices) == 0:
            print("NO Label")
            return pred.sum() * 0.0
        assert label in kept_indices, f"{label} not in {kept_indices}"
        map_dict = {old_ind: new_ind for new_ind, old_ind in enumerate(kept_indices)}
        pred = pred[:, kept_indices].mean(dim=0, keepdim=True)
        label = torch.zeros_like(pred[:, 0]).long() + map_dict[label]
        return F.cross_entropy(pred, label)

    def _single_coseg_loss(self, mask, fg_label, shape, bg_labels, cos_mat_dict):
        assert self.memory_bank_size == 1, \
            "MBS should be set to 1 for single-image co-segmentation!"
        h, w, H, W = shape
        # (h * w, 1) or (h * w, 2)
        coseg_score = self._coseg_score(fg_label, bg_labels, cos_mat_dict)
        coseg_score = self.coseg_head(coseg_score).reshape(H * W)
        mask = mask * coseg_score[:, None].sigmoid()
        coseg_loss = self._cross_entropy_loss(mask, fg_label, bg_labels)
        return coseg_loss

    def _single_coseg_bce_loss(self, mask, label, shape, cos_mat_dict):
        h, w, H, W, N = shape
        # (h * w, 1) or (h * w, 2)
        unique_label = torch.unique(label).flatten()
        unique_label = unique_label[unique_label != self.ignore_index].tolist()
        coseg_score_mat = torch.ones_like(mask)
        for fg_label in unique_label:
            bg_labels = [x for x in unique_label if x != fg_label]
            coseg_score = self._coseg_score(fg_label, bg_labels, cos_mat_dict)
            coseg_score = self.coseg_head(coseg_score).reshape(H * W)
            coseg_score_mat[:, fg_label] = coseg_score.sigmoid()
        mask = mask * coseg_score_mat
        # calculate bce loss
        valid = label != self.ignore_index
        label_one_hot = F.one_hot(label[valid], num_classes=N).float().sum(dim=0)
        label_one_hot = (label_one_hot > 0).int().float()
        if float(label_one_hot.sum()) > 0:
            pred = mask[valid].mean(dim=0)
            coseg_loss = F.binary_cross_entropy_with_logits(pred, label_one_hot)
        else:
            coseg_loss = mask.sum() * 0.0
        return coseg_loss

    def _single_pseudo_label_loss(self, seg_mask, seg_label, shape, cos_mat_dict):
        h, w, H, W, N = shape
        # (h * w, 1) or (h * w, 2)
        unique_label = torch.unique(seg_label).flatten()
        unique_label = unique_label[unique_label != self.ignore_index].tolist()
        if len(unique_label) == 0:
            return seg_mask.sum() * 0.0
        coseg_score_mat = torch.zeros_like(seg_mask) - 2.0
        for fg_label in unique_label:
            bg_labels = [x for x in unique_label if x != fg_label]
            coseg_score = self._coseg_score(fg_label, bg_labels, cos_mat_dict)
            coseg_score = self.coseg_head(coseg_score).reshape(H * W)
            coseg_score_mat[:, fg_label] = coseg_score.sigmoid()

        pseudo_label = coseg_score_mat.argmax(dim=1).flatten()
        pseudo_unique_label = torch.unique(pseudo_label).flatten()
        pseudo_unique_label = pseudo_unique_label[pseudo_unique_label != self.ignore_index].tolist()
        # print(f"{unique_label} != {pseudo_unique_label}", f"{coseg_score_mat.max(dim=0)}")
        assert len(set(unique_label) | set(pseudo_unique_label)) == len(set(unique_label)), \
            f"{unique_label} != {pseudo_unique_label}"
        coseg_loss = self.loss_decode(seg_mask, pseudo_label, ignore_index=self.ignore_index)
        return coseg_loss

    @torch.no_grad()
    def _coseg_score(self, fg_label, bg_labels, cos_mat_dict):
        fg_mat = cos_mat_dict[fg_label]
        if len(bg_labels):
            bg_mat = torch.stack([cos_mat_dict[i] for i in bg_labels], dim=0).mean(dim=0)
            coseg_score = fg_mat - bg_mat
        else:
            coseg_score = fg_mat
        return coseg_score.detach()

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
