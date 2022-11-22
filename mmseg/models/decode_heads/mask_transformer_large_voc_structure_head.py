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
class MaskTransformerLargeVocStructureHead(BaseDecodeHead):
    def __init__(
        self,
        n_cls,  # for evaluation
        patch_size,
        d_encoder=768,
        # head configs
        n_layers=3,
        n_heads=12,
        d_model=768,
        d_ff=3072,
        drop_path_rate=0.0,
        dropout=0.1,
        use_baseline=False,
        # other configs
        downsample_rate=8,
        temperature=1.0,
        # datasets
        all_cls_path="",
        mix_batch_datasets=["coco", "ade847"],
        test_dataset="ade847",
        ignore_indices=[255, -1],
        test_ignore_index=-1,
        # losses
        sample_num=512,
        prior_rate=1.0,
        prior_loss_weight=1.0,
        use_linear_classifier=False,
        use_auxiliary_loss=False,
        auxiliary_loss_weight=[],
        # weakly supervised
        weakly_supervised_datasets=["ade847"],
        weakly_prior_thresh=0.8,
        weakly_propagate_thresh=0.9,
        weakly_min_kept=1,
        weakly_max_kept=100,
        weakly_sample_num=5000,
        weakly_prior_loss_weight=1.0,
        weakly_structure_loss_weight=0.0,
        # structure loss
        structure_loss_weight=1.0,
        structure_loss_thresh=0.2,
        structure_branch_use_prior_loss=True,
        structure_branch_detach=False,
        structure_memory_bank_size=10,
        structure_sample_neg_class=50,
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
        self.n_layers = n_layers
        self.d_encoder = d_encoder
        self.n_cls = n_cls
        self.d_model = d_model
        self.scale = d_model**-0.5
        self.downsample_rate = downsample_rate
        self.temperature = temperature
        # process datasets, valid for only one dataset
        if os.path.exists(all_cls_path):
            self.all_cls = json.load(open(all_cls_path))
        else:
            self.all_cls = None
        self.mix_batch_datasets = mix_batch_datasets
        self.test_dataset = test_dataset
        self.ignore_indices = ignore_indices
        self.test_ignore_index = test_ignore_index
        # prior loss
        self.sample_num = sample_num
        assert prior_rate <= 1.0
        self.prior_rate = prior_rate
        self.prior_loss_weight = prior_loss_weight
        self.use_linear_classifier = use_linear_classifier
        self.use_auxiliary_loss = use_auxiliary_loss
        self.auxiliary_loss_weight = (
            auxiliary_loss_weight if use_auxiliary_loss else [1.0]
        )
        # weakly supervised
        self.weakly_supervised_datasets = weakly_supervised_datasets
        self.weakly_prior_thresh = weakly_prior_thresh
        self.weakly_propagate_thresh = weakly_propagate_thresh
        self.weakly_min_kept = weakly_min_kept
        self.weakly_max_kept = weakly_max_kept
        self.weakly_sample_num = weakly_sample_num
        self.weakly_prior_loss_weight = weakly_prior_loss_weight
        self.weakly_structure_loss_weight = weakly_structure_loss_weight
        # structure loss
        self.structure_loss_weight = structure_loss_weight
        self.structure_loss_thresh = structure_loss_thresh
        self.structure_memory_bank_size = structure_memory_bank_size
        self.structure_sample_neg_class = structure_sample_neg_class
        # oracle experiment
        self.oracle_inference = oracle_inference
        self.num_oracle_points = num_oracle_points
        self.oracle_downsample_rate = oracle_downsample_rate
        # propagation head
        self.structure_branch_use_prior_loss = structure_branch_use_prior_loss
        self.structure_branch_detach = structure_branch_detach
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks_c = nn.ModuleList(
            [
                Block(
                    dim=d_model,
                    heads=n_heads,
                    mlp_dim=d_ff,
                    dropout=dropout,
                    drop_path=dpr[i],
                )
                for i in range(n_layers)
            ]
        )
        self.blocks_s = nn.ModuleList(
            [
                Block(
                    dim=d_model,
                    heads=n_heads,
                    mlp_dim=d_ff,
                    dropout=dropout,
                    drop_path=dpr[i],
                )
                for i in range(n_layers)
            ]
        )
        num_layer = n_layers if self.use_auxiliary_loss else 1
        self.num_layer = num_layer
        self.norm_c = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layer)])
        self.norm_s = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layer)])
        # cosine classifier
        num_classes = self.num_classes if self.all_cls is None else len(self.all_cls)
        self.gamma_c = nn.Parameter(torch.ones(num_layer))
        self.beta_c = nn.Parameter(torch.zeros([num_layer]))
        self.proj_feat_c = nn.Parameter(
            self.scale * torch.randn(num_layer, d_model, d_model)
        )
        self.proj_cls_c = nn.Parameter(
            self.scale * torch.randn(num_layer, d_model, d_model)
        )
        self.cls_emb_c = nn.Parameter(torch.randn(num_classes, d_model))
        if self.structure_branch_use_prior_loss:
            self.gamma_s = nn.Parameter(torch.ones(num_layer))
            self.beta_s = nn.Parameter(torch.zeros(num_layer))
            self.proj_feat_s = nn.Parameter(
                self.scale * torch.randn(num_layer, d_model, d_model)
            )
            self.proj_cls_s = nn.Parameter(
                self.scale * torch.randn(num_layer, d_model, d_model)
            )
            self.cls_emb_s = nn.Parameter(torch.randn(num_classes, d_model))
        # memory bank
        for i in range(num_classes):
            self.register_buffer(
                "queue" + str(i), torch.randn(self.structure_memory_bank_size, d_model)
            )
            self.register_buffer("ptr" + str(i), torch.zeros(1, dtype=torch.long))
            exec(
                "self.queue"
                + str(i)
                + "="
                + "nn.functional.normalize("
                + "self.queue"
                + str(i)
                + ",dim=0)"
            )
        self.all_num_classes = num_classes
        self.prior_ind_bucket = None

    def init_weights(self):
        self.apply(init_weights)
        if not self.use_linear_classifier:
            trunc_normal_(self.cls_emb_c, std=0.02)
            if self.structure_branch_use_prior_loss:
                trunc_normal_(self.cls_emb_s, std=0.02)

    def _dequeue_and_enqueue(self, feat, cls):
        """
        Params:
            feat: torch.Tensor(d_model)
            cls: int, class index of current dataset
        """
        cls_ind = self.cls_index[cls]
        ptr = int(eval("self.ptr" + str(cls_ind)))
        eval("self.queue" + str(cls_ind))[ptr] = feat
        ptr = (ptr + 1) % self.structure_memory_bank_size
        eval("self.ptr" + str(cls_ind))[0] = ptr

    def _update(self, training):
        rank, _ = get_dist_info()
        if training:
            self.dataset_on_gpu = self.mix_batch_datasets[
                rank % len(self.mix_batch_datasets)
            ]
            self.ignore_index = self.ignore_indices[rank % len(self.mix_batch_datasets)]
            if self.dataset_on_gpu in self.weakly_supervised_datasets:
                self.prior_loss_weight = self.weakly_prior_loss_weight
                self.structure_loss_weight = self.weakly_structure_loss_weight
                self.sample_num = self.weakly_sample_num
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

        if self.all_cls is not None:
            self.cls_index = [self.all_cls.index(name) for name in cls_name]
        else:
            self.cls_index = list(range(len(cls_name)))

    def get_mask(self, feat, cls_emb, proj_feat, proj_cls, gamma, beta, shape):
        B, C, H, W = shape
        cls_emb = cls_emb.expand(shape[0], -1, -1)
        cls_emb = cls_emb @ proj_cls
        cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
        feat = feat @ proj_feat
        feat = feat / feat.norm(dim=-1, keepdim=True)
        masks = feat @ cls_emb.transpose(1, 2)
        cos_mat = masks.clone().detach()
        masks = (
            (masks - torch.mean(masks, dim=-1, keepdim=True))
            / torch.sqrt(torch.var(masks, dim=-1, keepdim=True, unbiased=False) + 1e-5)
        ) * gamma + beta
        masks = masks.view(B, H, W, self.all_num_classes).permute(0, 3, 1, 2)
        cos_mat = cos_mat.view(B, H, W, self.all_num_classes).permute(0, 3, 1, 2)
        return masks, cos_mat

    def forward(self, x, img_metas, img_labels=None):
        x = self._transform_inputs(x)
        B, C, H, W = x.size()
        x = x.view(B, C, -1).permute(0, 2, 1)
        fc = fs = x
        if self.structure_branch_detach:
            fs = fs.detach()
        fc_list, fs_list = [], []
        for blk_c, blk_s in zip(self.blocks_c, self.blocks_s):
            fc, fs = blk_c(fc), blk_s(fs)
            fc_list.append(fc)
            fs_list.append(fs)
        if not self.use_auxiliary_loss:
            fc_list = [fc_list[-1]]
            fs_list = [fs_list[-1]]
        embeds = fs.clone().view(B, H, W, -1).permute(0, 3, 1, 2)
        masks_c_list, masks_s_list = [], []
        cos_mat_c_list, cos_mat_s_list = [], []
        for lid in range(self.num_layer):
            fc = self.norm_c[lid](fc_list[lid])
            fs = self.norm_s[lid](fs_list[lid])
            # generate classification masks
            masks_c, cos_mat_c = self.get_mask(
                feat=fc,
                cls_emb=self.cls_emb_c,
                proj_feat=self.proj_feat_c[lid],
                proj_cls=self.proj_cls_c[lid],
                gamma=self.gamma_c[lid],
                beta=self.beta_c[lid],
                shape=(B, C, H, W),
            )
            masks_c_list.append(masks_c)
            cos_mat_c_list.append(cos_mat_c)
            # generate masks by structuring feature
            if not self.structure_branch_use_prior_loss:
                continue
            masks_s, cos_mat_s = self.get_mask(
                feat=fs,
                cls_emb=self.cls_emb_s,
                proj_feat=self.proj_feat_s[lid],
                proj_cls=self.proj_cls_s[lid],
                gamma=self.gamma_s[lid],
                beta=self.beta_s[lid],
                shape=(B, C, H, W),
            )
            masks_s_list.append(masks_s)
            cos_mat_s_list.append(cos_mat_s)
        return masks_c_list, masks_s_list, cos_mat_c_list, cos_mat_s_list, embeds

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        self._update(training=True)
        masks_c, masks_s, cos_mat_c, cos_mat_s, embeds = self.forward(inputs, img_metas)
        losses = dict()
        for lid in range(self.num_layer):
            loss = self.losses(
                masks_c[lid],
                masks_s[lid],
                cos_mat_c[lid],
                cos_mat_s[lid],
                embeds,
                gt_semantic_seg,
            )
            weight = self.auxiliary_loss_weight[lid]
            for k, v in loss.items():
                v = v * weight if "loss" in k else v
                losses.update({f"{lid}_" + k: v})
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg=None):
        self._update(training=False)
        masks_c, _, _, _, embeds = self.forward(inputs, img_metas)
        masks_c = masks_c[-1][:, self.cls_index]
        if self.oracle_inference:
            assert gt_semantic_seg is not None
            masks_c = self.oracle_propagation(embeds, gt_semantic_seg)
        return masks_c

    @force_fp32(apply_to=("seg_mask", "seg_mask_s"))
    def losses(self, seg_mask, seg_mask_s, cos_mat, cos_mat_s, seg_embed, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        h = seg_label.shape[-2] // self.downsample_rate
        w = seg_label.shape[-1] // self.downsample_rate
        # (256, 256), (160, 160)
        seg_mask = resize(
            seg_mask, size=(h, w), mode="bilinear", align_corners=self.align_corners
        )
        cos_mat = resize(
            cos_mat, size=(h, w), mode="bilinear", align_corners=self.align_corners
        )
        seg_embed = resize(
            seg_embed, size=(h, w), mode="bilinear", align_corners=self.align_corners
        )
        seg_label = resize(seg_label.float(), size=(h, w), mode="nearest").long()

        ########## classification task ##########
        loss["loss_classification"] = (
            self.loss_classification(seg_mask, cos_mat, seg_embed, seg_label)
            * self.prior_loss_weight
        )
        # add classification supervision to structuring branch
        if self.structure_branch_use_prior_loss:
            seg_mask_s = resize(
                seg_mask_s,
                size=(h, w),
                mode="bilinear",
                align_corners=self.align_corners,
            )
            cos_mat_s = resize(
                cos_mat_s,
                size=(h, w),
                mode="bilinear",
                align_corners=self.align_corners,
            )
            loss["loss_classification_s"] = (
                self.loss_classification(seg_mask_s, cos_mat_s, seg_embed, seg_label)
                * self.prior_loss_weight
            )
        # log accuracy
        loss["acc_seg"] = self.log_accuracy(seg_mask, seg_label)

        ########## structuring task ##########
        loss["loss_structuring"] = (
            self.loss_structuring(seg_embed, seg_label) * self.structure_loss_weight
        )
        return loss

    def log_accuracy(self, seg_mask, seg_label):
        B, N, H, W = seg_mask.shape
        seg_label = seg_label.flatten()
        seg_label_ = copy.deepcopy(seg_label)
        unique_label = torch.unique(seg_label)
        unique_label = unique_label[unique_label != self.ignore_index]
        for l in unique_label:
            seg_label_[seg_label == l] = self.cls_index[int(l)]
        seg_mask = seg_mask.permute(0, 2, 3, 1).reshape(B * H * W, N)
        imagenet_in_batch = any(["in" in x for x in self.mix_batch_datasets])
        acc_weight = 0.0 if "in" in self.dataset_on_gpu else 2.0
        acc_weight = acc_weight if imagenet_in_batch else 1.0
        return accuracy(seg_mask, seg_label_)

    def _sample(self, seg_mask, seg_label, min_kept=1):
        """
        Args:
            seg_mask (torch.Tensor): segmentation logits, shape (B, N, H, W)
            seg_label (torch.Tensor): segmentation label, shape (B, 1, H, W)

        Returns:
            pos_bucket
            prior_bucket
        """
        B, N, H, W = seg_mask.size()
        seg_label = seg_label.reshape(B * H * W)
        seg_mask = seg_mask.permute(0, 2, 3, 1).reshape(B * H * W, N)
        prior_mask = seg_mask
        prior_label = copy.deepcopy(seg_label)
        unique_label = torch.unique(seg_label)
        unique_label = unique_label[unique_label != self.ignore_index]
        for l in unique_label:
            prior_label[seg_label == l] = self.cls_index[int(l)]
        return prior_mask, prior_label

    def _weak_sample(self, seg_mask, cos_mat, seg_embed, seg_label):
        """
        for each image, smaple each category separately
        i.e. a pixel could be assigned to more than one categories
        """
        seg_embed = seg_embed.detach()
        B, N, H, W = seg_mask.size()
        prior_mask, prior_label = [], []
        prior_ind_bucket = {}
        for b, (mask, cos, embed, label) in enumerate(
            zip(seg_mask, cos_mat, seg_embed, seg_label)
        ):
            mask = mask.reshape(N, H * W).permute(1, 0)
            cos = cos.reshape(N, H * W).permute(1, 0)
            embed = embed.reshape(-1, H * W).permute(1, 0)
            label = label.reshape(H * W)
            unique_label = torch.unique(label)
            unique_label = unique_label[unique_label != self.ignore_index]
            for l in unique_label:
                l = self.cls_index[int(l)]
                inds = (
                    (cos[:, l] > self.weakly_prior_thresh)
                    .nonzero(as_tuple=False)
                    .flatten()
                )
                if inds.numel() < self.weakly_min_kept:
                    inds = cos[:, l].topk(self.weakly_min_kept).indices
                elif inds.numel() > self.weakly_max_kept:
                    inds = cos[:, l].topk(self.weakly_max_kept).indices
                # propogate seed
                embed = embed / embed.norm(dim=-1, keepdim=True)
                prop_mat = embed @ embed[inds].T
                prop_inds = (
                    (prop_mat.max(dim=-1).values > self.weakly_propagate_thresh)
                    .nonzero(as_tuple=False)
                    .flatten()
                )
                inds = torch.cat([inds, prop_inds], dim=0).unique()
                prior_mask.append(mask[inds])
                prior_label.append(torch.ones_like(label[inds]) * l)
                prior_ind = inds + b * H * W
                if l not in prior_ind_bucket:
                    prior_ind_bucket[l] = []
                prior_ind_bucket[l].append(prior_ind)
        prior_mask = torch.cat(prior_mask, dim=0)
        prior_label = torch.cat(prior_label, dim=0)
        self.prior_ind_bucket = {
            k: torch.cat(v, dim=0) for k, v in prior_ind_bucket.items()
        }
        return prior_mask, prior_label

    def loss_classification(self, seg_mask, cos_mat, seg_embed, seg_label):
        if self.dataset_on_gpu in self.weakly_supervised_datasets:
            prior_mask, prior_label = self._weak_sample(
                seg_mask, cos_mat, seg_embed, seg_label
            )
        else:
            prior_mask, prior_label = self._sample(seg_mask, seg_label)
        # no valid label
        if prior_mask is None:
            return seg_mask.sum() * 0.0
        else:
            assert prior_label is not None
            return self.loss_decode(
                prior_mask, prior_label, weight=None, ignore_index=self.ignore_index
            )

    def loss_structuring(self, seg_embed, seg_label):
        # turn off weakly structure loss
        if self.structure_loss_weight == 0.0:
            return seg_embed.sum() * 0.0
        # bucket for each class in the batch
        B, C, H, W = seg_embed.shape
        seg_embed = seg_embed.permute(0, 2, 3, 1).reshape(B * H * W, C)
        seg_label = seg_label.reshape(B * H * W)
        unique_label = torch.unique(seg_label)
        unique_label = unique_label[unique_label != self.ignore_index]
        if self.dataset_on_gpu in self.weakly_supervised_datasets:
            # assert False, f"{self.prior_ind_bucket}, {unique_label}"
            bucket = [
                self.prior_ind_bucket[self.cls_index[int(l)]] for l in unique_label
            ]
        else:
            bucket = [torch.nonzero(seg_label == l)[:, 0] for l in unique_label]
        if len(bucket) == 0:
            return seg_embed[seg_label != self.ignore_index].sum()
        inds = self._sample_random_points(bucket)
        # positive and negative samples
        loss = 0
        for l, ind in zip(unique_label, inds):
            l_ind = self.cls_index[int(l)]
            embed = seg_embed[ind]
            pos_embed = eval("self.queue" + str(l_ind)).clone().detach()

            class_inds = list(range(self.all_num_classes))
            class_inds.remove(l_ind)
            neg_inds = np.random.choice(class_inds, size=50, replace=False)
            neg_embed_list = []
            for neg_ind in neg_inds:
                neg_embed_list.append(
                    eval("self.queue" + str(neg_ind)).clone().detach()
                )
            neg_embed = torch.cat(neg_embed_list, dim=0)

            # calculate loss
            embed = embed / embed.norm(dim=-1, keepdim=True)
            pos_embed = pos_embed / pos_embed.norm(dim=-1, keepdim=True)
            neg_embed = neg_embed / neg_embed.norm(dim=-1, keepdim=True)
            cos_mat = embed @ torch.cat([embed, pos_embed, neg_embed], dim=0).transpose(
                0, 1
            )

            label = torch.ones_like(embed[:, 0])
            pos_label = torch.ones_like(pos_embed[:, 0])
            neg_label = torch.zeros_like(neg_embed[:, 0])
            label_mat = (
                (label[:, None] == torch.cat([label, pos_label, neg_label])[None, :])
                .int()
                .float()
            )

            mask_ = (cos_mat > self.structure_loss_thresh) | (label_mat == 1)
            loss += torch.pow(cos_mat[mask_] - label_mat[mask_], 2).mean()
        # dequeue and enqueue
        for l, ind in zip(unique_label, inds):
            i = int(ind[int(np.random.choice(range(len(ind)), size=1))])
            self._dequeue_and_enqueue(seg_embed[i], int(l))
        return loss

    def _sample_random_points(self, buckets):
        """Sample points from each buckets
        Args:
            num_per_buckets (list): number of points in each class
        """
        num_per_buckets = [len(p) for p in buckets]
        sample_per_bucket = [
            self.sample_num // len(buckets) for _ in range(len(num_per_buckets))
        ]
        if len(sample_per_bucket) > 1:
            sample_per_bucket[-1] = self.sample_num - sum(sample_per_bucket[:-1])
        else:
            sample_per_bucket[0] = self.sample_num
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
            mode="bilinear",
            align_corners=self.align_corners,
        )
        seg_label = resize(input=seg_label.float(), size=(h, w), mode="nearest").long()[
            0, 0
        ]
        if self.dataset_on_gpu == "ade150":
            seg_label = seg_label - 1
            seg_label[seg_label == -1] = self.ignore_index
        seg_embed = seg_embed.permute(0, 2, 3, 1)
        seg_label_per_image = seg_label.reshape(h * w)
        seg_embed_per_image = seg_embed.reshape(h * w, C)
        seg_embed_per_image = seg_embed_per_image / seg_embed_per_image.norm(
            dim=-1, keepdim=True
        )
        unique_label = torch.unique(seg_label_per_image)
        unique_label = unique_label[unique_label != self.ignore_index]
        masks = torch.zeros((B, self.num_classes, h, w), device=device)
        for l in unique_label:
            pos_inds = (seg_label_per_image == l).nonzero(as_tuple=False)[:, 0]
            inds = torch.randperm(len(pos_inds))[: self.num_oracle_points]
            prior_inds = pos_inds[inds]
            cos_mat = seg_embed_per_image[prior_inds] @ seg_embed_per_image.T
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

    def forward(self, x):
        B, _, C = x.shape
        # B, head, N, C // head
        q = (
            self.q_linear(x)
            .reshape(B, -1, self.heads, C // self.heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_linear(x)
            .reshape(B, -1, self.heads, C // self.heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_linear(x)
            .reshape(B, -1, self.heads, C // self.heads)
            .permute(0, 2, 1, 3)
        )

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
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
