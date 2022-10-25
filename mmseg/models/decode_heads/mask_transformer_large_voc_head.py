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
        use_attn_head=False,
        seed_rate=1.0,
        adaptive_seed_score=False,
        temperature=1.0,
        # datasets
        all_cls_path="",
        mix_batch_datasets=["coco", "ade847"],
        test_dataset="ade847",
        ignore_indices=[255, -1],
        test_ignore_index=-1,
        # weakly supervised
        weakly_supervised_datasets=["ade847"],
        weakly_seed_thresh=0.8,
        weakly_prior_thresh=0.9,
        weakly_min_kept=1,
        weakly_max_kept=100,
        weakly_seed_loss_weight=1.0,
        weakly_use_avgpool=False,
        # contrastive loss
        use_structure_loss=False,
        structure_loss_weight=1.0,
        structure_loss_thresh=0.2,
        # memory bank
        use_memory_bank=False,
        memory_bank_size=50,
        coseg_weight=1.0,
        coseg_use_mean=False,
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
        self.seed_rate = seed_rate
        self.adaptive_seed_score = adaptive_seed_score
        self.use_attn_head = use_attn_head
        # weakly supervised
        self.weakly_supervised_datasets = weakly_supervised_datasets
        self.weakly_seed_thresh = weakly_seed_thresh
        self.weakly_min_kept = weakly_min_kept
        self.weakly_max_kept = weakly_max_kept
        self.weakly_seed_loss_weight = weakly_seed_loss_weight
        self.weakly_use_avgpool = weakly_use_avgpool
        # contrastive loss
        self.use_structure_loss = use_structure_loss
        self.structure_loss_weight = structure_loss_weight
        self.structure_loss_thresh = structure_loss_thresh
        # oracle experiment
        self.oracle_inference = oracle_inference
        self.num_oracle_points = num_oracle_points
        self.oracle_downsample_rate = oracle_downsample_rate
        # visualization
        self.visualize_seed = visualize_seed
        self.visualize_out_dir = visualize_out_dir

        # self.proj_dec = nn.Linear(d_encoder, d_model)
        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        # attention head
        if self.use_attn_head:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
            self.blocks = nn.ModuleList([
                Block(dim=d_model, heads=n_heads, mlp_dim=d_ff, dropout=dropout, drop_path=dpr[i])
                for i in range(n_layers)
            ])
            self.decoder_norm = nn.LayerNorm(d_model)
        # cosine classifier
        self.gamma = nn.Parameter(torch.ones([]))
        self.beta = nn.Parameter(torch.zeros([]))
        self.all_classes = self.num_classes if self.all_cls is None else len(self.all_cls)
        self.cls_emb = nn.Parameter(torch.randn(self.all_classes, d_model))
        # memory bank
        self.use_memory_bank = use_memory_bank
        self.memory_bank_size = memory_bank_size
        self.coseg_weight = coseg_weight
        self.coseg_use_mean = coseg_use_mean
        if self.use_memory_bank:
            self.register_buffer(f"queue", torch.randn(self.all_classes, self.memory_bank_size, self.d_model))
            self.register_buffer(f"ptr", torch.zeros(self.all_classes, dtype=torch.long))
            self.queue = self.queue / self.queue.norm(dim=-1, keepdim=True)

    def init_weights(self):
        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    def _dequeue_and_enqueue(self, feat, cls):
        """
        Params:
            feat: torch.Tensor(d_model)
            cls: int, class index of current dataset
        """
        cls_ind = self.cls_index[cls]
        ptr = int(self.ptr[cls_ind])
        self.queue[cls_ind, ptr] = feat
        ptr = (ptr + 1) % self.memory_bank_size
        self.ptr[cls_ind] = ptr

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
        # x = self.proj_dec(x)
        cls_emb = self.cls_emb[self.cls_index]
        cls_emb = cls_emb.expand(x.size(0), -1, -1)
        if self.use_attn_head:
            for blk in self.blocks:
                x = blk(x)
            patches = self.decoder_norm(x)
        else:
            patches = x
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
        masks, embeds, scores = self.forward(inputs, img_metas)
        if self.visualize_seed:
            self.visualize_imagenet(img, masks, scores, embeds, gt_semantic_seg, img_metas)
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
        seg_label = resize(
            seg_label.float(), size=(h, w), mode='nearest'
        ).long()

        # classification task
        B, N, H, W = seg_mask.shape
        if self.dataset_on_gpu in self.weakly_supervised_datasets:
            seed_mask, seed_label = self._weak_sample(seg_mask, seg_score, seg_embed, seg_label)
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
            seg_embed = resize(
                seg_embed, size=(h, w), mode='bilinear', align_corners=self.align_corners
            )
            loss['loss_structure'] = self.loss_structure(
                seg_embed, seg_label
            ) * self.structure_loss_weight
        return loss
    
    def log_accuracy(self, seg_mask, seg_label):
        B, N, H, W = seg_mask.shape
        seg_label = seg_label.flatten()
        seg_mask = seg_mask.permute(0, 2, 3, 1).reshape(B * H * W, N)
        weak_in_batch = len(self.weakly_supervised_datasets) > 0
        acc_weight = 0.0 if self.dataset_on_gpu in self.weakly_supervised_datasets else 2.0
        acc_weight = acc_weight if weak_in_batch else 1.0
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
            k = int(len(p) * self.seed_rate)
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

    def _weak_sample(self, seg_mask, seg_score, seg_embed, seg_label):
        """
        for each image, smaple each category separately
        i.e. a pixel could be assigned to more than one categories
        """
        B, N, H, W = seg_mask.size()
        B, C, h, w = seg_embed.size()
        seg_embed2 = resize(
            seg_embed, size=(H, W), mode='bilinear', align_corners=self.align_corners
        ).clone().detach()
        seed_mask, seed_label = [], []
        # assert False, f"{seg_label.unique()}"
        for mask, score, embed, embed2, label in zip(
            seg_mask, seg_score, seg_embed, seg_embed2, seg_label
        ):
            mask = mask.reshape(N, H * W).permute(1, 0)
            label = label.reshape(H * W)
            score = score.reshape(N, H * W).permute(1, 0)
            embed = embed.reshape(C, h * w).permute(1, 0)
            embed2 = embed2.reshape(C, H * W).permute(1, 0)
            unique_label = torch.unique(label)
            unique_label = unique_label[unique_label != self.ignore_index]
            # NOTE: only support imagenet now
            if self.weakly_use_avgpool:
                assert len(unique_label) == 1
                seed_mask.append(mask.mean(dim=0, keepdim=True))
                seed_label.append(torch.ones_like(label[[0]]) * int(unique_label))
                continue
            for l in unique_label:
                l = int(l)
                label_score = score[:, l]
                if self.use_memory_bank:
                    coseg_score = self.cross_image_score(embed, l, (h, w, H, W))
                    label_score = label_score + coseg_score * self.coseg_weight
                weakly_seed_thresh = self.weakly_seed_thresh
                if self.adaptive_seed_score:
                    weakly_seed_thresh = label_score.mean() + label_score.std()
                inds = (label_score > weakly_seed_thresh).nonzero(as_tuple=False).flatten()
                if inds.numel() < self.weakly_min_kept:
                    inds = label_score.topk(self.weakly_min_kept).indices
                elif inds.numel() > self.weakly_max_kept:
                    inds = label_score.topk(self.weakly_max_kept).indices
                seed_mask.append(mask[inds])
                seed_label.append(torch.ones_like(label[inds]) * l)
                if self.use_memory_bank:
                    region_embed = embed2[inds].mean(dim=0).clone().detach()
                    region_embed = region_embed / region_embed.norm(dim=-1, keepdim=True)
                    self._dequeue_and_enqueue(feat=region_embed, cls=l)
        
        seed_mask = torch.cat(seed_mask, dim=0)
        seed_label = torch.cat(seed_label, dim=0)
        return seed_mask, seed_label
    
    def cross_image_score(self, embed, label, shape):
        # NOTE: we should use global label index to fetch the correct feature
        # This line fix a bug!
        label = self.cls_index[label]
        h, w, H, W = shape
        cross_embed = self.queue[label].clone().detach()
        cross_embed = cross_embed / cross_embed.norm(dim=-1, keepdim=True)
        embed = embed / embed.norm(dim=-1, keepdim=True)
        coseg_score = embed @ cross_embed.T
        coseg_score = coseg_score.reshape(h, w, -1).permute(2, 0, 1)
        coseg_score = resize(
            coseg_score[None], size=(H, W), mode='bilinear', align_corners=self.align_corners
        )[0].reshape(-1, H * W)
        # NOTE: should we change this to mean?
        if self.coseg_use_mean:
            coseg_score = coseg_score.mean(dim=0)
        else:
            coseg_score = coseg_score.max(dim=0).values
        # rank, _ = get_dist_info()
        # if rank == 0:
        #     print("coseg_score", coseg_score.mean())
        return coseg_score
    
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

    def visualize_imagenet(self, img, mask, score, embed, label, img_metas):
        # h = label.shape[-2]
        # w = label.shape[-1]
        # img = img[0]
        # mask = resize(
        #     mask, size=(h, w), mode='bilinear', align_corners=self.align_corners
        # )[0]
        # score = resize(
        #     score, size=(h, w), mode='bilinear', align_corners=self.align_corners
        # )[0]
        # embed = resize(
        #     embed, size=(h, w), mode='bilinear', align_corners=self.align_corners
        # )[0]
        # # self._weak_sample()
        # N, H, W = mask.shape
        # mask = mask.reshape(N, H * W).permute(1, 0)
        # label = label.reshape(H * W)
        rank, _ = get_dist_info()
        if rank == 0:
            unique_label = torch.unique(label)
            unique_label = unique_label[unique_label != self.ignore_index]
            assert len(unique_label) == 1
            l = int(unique_label)
            # png_name = self.cls_name[l]+"_"+img_metas[0]["ori_filename"].split("/")[-1].replace("JPEG", "png")   
            # pth_name = self.cls_name[l]+"_"+img_metas[0]["ori_filename"].split("/")[-1].replace("JPEG", "pth")
            pth_name = "notebook/cls_emb130.pth"
            # os.makedirs(, exist_ok=True)
            torch.save(self.cls_emb[self.cls_index], pth_name)
        exit()
        # torch.save(embed, os.path.join(self.visualize_out_dir, pth_name))
        # for i in range(3):
        #     img[i] = (img[i] - img[i].min()) / (img[i].max() - img[i].min())
        # # (H, W, 3)
        # img = img.permute(1, 2, 0)
        # thresh_fg = score[l].flatten().topk(5000).values.min()
        # thresh_bg = score[l].flatten().topk(5000, largest=False).values.min()
        # print((score[l] >= thresh_fg).sum())
        # fg_mask = score[l].flatten() >= thresh_fg
        # bg_mask = score[l].flatten() <= thresh_bg
        # embed = embed.flatten(1).permute(1,0)
        # cross_embed = self.queue[self.cls_index[l]].clone().detach()
        # cross_score = embed @ cross_embed.T
        # mode = "mean" # "max"
        # if mode == "mean":
        #     coseg_score = cross_score.mean(dim=-1).flatten()
        # elif mode == "max":
        #     coseg_score = cross_score.max(dim=-1).values.flatten()
        # inds = (coseg_score > (coseg_score.mean() + coseg_score.std())).nonzero(as_tuple=False).flatten()
        # min_kept = 4000
        # max_kept = 40000
        # if inds.numel() < min_kept:
        #     inds = coseg_score.topk(min_kept).indices
        # elif inds.numel() > max_kept:
        #     inds = coseg_score.topk(max_kept).indices
        # print(len(inds))
        # score = score[l].flatten()
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # os.makedirs(self.visualize_out_dir, exist_ok=True)
        # sns.kdeplot(score.cpu().numpy(), bw_adjust=0.2)
        # for i in range(10):
        #     sns.kdeplot(cross_score[i].cpu().numpy(), bw_adjust=0.2)
        # plt.savefig(os.path.join(self.visualize_out_dir, png_name))
        # plt.close()
        # score = score[l].flatten()
        # inds = (score > (score.mean() + score.std())).nonzero(as_tuple=False).flatten()
        # min_kept = 4000
        # max_kept = 40000
        # if inds.numel() < min_kept:
        #     inds = score.topk(min_kept).indices
        # elif inds.numel() > max_kept:
        #     inds = score.topk(max_kept).indices
        # print(len(inds))
        # coseg_thresh = coseg_score.flatten().topk(40000).values.min()
        # print(cos_mat.shape)
        # background_cos_mat = embed[bg_mask] @ embed.T
        # foreground_score = foreground_cos_mat.mean(dim=0).reshape(H, W)
        # background_score = background_cos_mat.mean(dim=0).reshape(H, W)
        # print(foreground_score.shape, background_score.shape)
        
        # print(score.max(), score.min(), score.std())
        # score_img = torch.zeros_like(img).flatten(0,1)
        # score_img[inds] = 1.0
        # score_img = score_img.reshape(*img.shape)
        # score_img[foreground_score > 0.6] = 1.0
        # score_img[:,:,1] = score
        # score_img[:,:,2] = score
        # img = img * 0.5 + score_img * 0.5
        # img = img.cpu().numpy()
        # img = (img * 255).astype(np.uint8)
        # os.makedirs(self.visualize_out_dir, exist_ok=True)
        # Image.fromarray(img).save(os.path.join(self.visualize_out_dir, png_name), format="PNG")
    
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