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
class MaskTransformerLargeVocPropagationHead(BaseDecodeHead):
    def __init__(
        self,
        n_cls,  # for evaluation
        patch_size,
        d_encoder,
        # propagation head
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
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
        use_auxiliary_prior_loss=False,
        # weakly supervised
        weakly_supervised_datasets=["ade847"],
        weakly_prior_thresh=0.8,
        weakly_min_kept=1,
        weakly_max_kept=100,
        weakly_sample_num=5000,
        weakly_pos_sample_num=10,
        weakly_neg_sample_num=10,
        weakly_prior_loss_weight=1.0,
        weakly_structure_pos_thresh=0.9,
        weakly_structure_neg_thresh=0.5,
        weakly_structure_loss_weight=0.0,
        # structure loss
        structure_loss_weight=1.0,
        structure_loss_thresh=0.2,
        structure_loss_no_negative=False,
        structure_branch_use_prior_loss=True,
        structure_branch_separate_classifier=False,
        structure_branch_detach=False,
        structure_gamma_initial_value=1.0,
        structure_qk_cosine=False,
        structure_inference=False,
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
        self.use_auxiliary_prior_loss = use_auxiliary_prior_loss
        # weakly supervised
        self.weakly_supervised_datasets = weakly_supervised_datasets
        self.weakly_prior_thresh = weakly_prior_thresh
        self.weakly_min_kept = weakly_min_kept
        self.weakly_max_kept = weakly_max_kept
        self.weakly_sample_num = weakly_sample_num
        self.weakly_pos_sample_num = weakly_pos_sample_num
        self.weakly_neg_sample_num = weakly_neg_sample_num
        self.weakly_prior_loss_weight = weakly_prior_loss_weight
        self.weakly_structure_pos_thresh = weakly_structure_pos_thresh
        self.weakly_structure_neg_thresh = weakly_structure_neg_thresh
        self.weakly_structure_loss_weight = weakly_structure_loss_weight
        # structure loss
        self.structure_loss_weight = structure_loss_weight
        self.structure_loss_thresh = structure_loss_thresh
        self.structure_loss_no_negative = structure_loss_no_negative
        self.structure_inference = structure_inference
        # oracle experiment
        self.oracle_inference = oracle_inference
        self.num_oracle_points = num_oracle_points
        self.oracle_downsample_rate = oracle_downsample_rate
        # visualization
        self.visualize_seed = visualize_seed
        self.visualize_out_dir = visualize_out_dir

        # self.proj_dec = nn.Linear(d_encoder, d_model)
        # propagation head
        self.structure_branch_use_prior_loss = structure_branch_use_prior_loss
        self.structure_branch_separate_classifier = structure_branch_separate_classifier
        self.structure_branch_detach = structure_branch_detach
        self.structure_gamma_initial_value = structure_gamma_initial_value
        self.structure_qk_cosine = structure_qk_cosine
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=d_model,
                    heads=n_heads,
                    mlp_dim=d_ff,
                    dropout=dropout,
                    drop_path=dpr[i],
                    gamma_init=self.structure_gamma_initial_value,
                    use_baseline=use_baseline,
                )
                for i in range(n_layers)
            ]
        )
        self.norm_c = nn.LayerNorm(d_model)
        self.norm_s = nn.LayerNorm(d_model)
        # cosine classifier
        self.gamma_c = nn.Parameter(torch.ones([]))
        self.beta_c = nn.Parameter(torch.zeros([]))
        if self.structure_branch_use_prior_loss:
            self.gamma_s = nn.Parameter(torch.ones([]))
            self.beta_s = nn.Parameter(torch.zeros([]))
        num_classes = self.num_classes if self.all_cls is None else len(self.all_cls)
        if self.use_linear_classifier:
            self.cls_emb_c = nn.Linear(d_model, num_classes)
            self.cls_emb_s = None
            if (
                self.structure_branch_use_prior_loss
            ):  # and self.structure_branch_separate_classifier:
                self.cls_emb_s = nn.Linear(d_model, num_classes)
        else:
            self.proj_feat_c = nn.Parameter(self.scale * torch.randn(d_model, d_model))
            self.proj_cls_c = nn.Parameter(self.scale * torch.randn(d_model, d_model))
            self.cls_emb_c = nn.Parameter(torch.randn(num_classes, d_model))
            if self.structure_branch_use_prior_loss:
                self.proj_feat_s = nn.Parameter(
                    self.scale * torch.randn(d_model, d_model)
                )
                self.proj_cls_s = nn.Parameter(
                    self.scale * torch.randn(d_model, d_model)
                )
            self.cls_emb_s = None
            if (
                self.structure_branch_use_prior_loss
            ):  # and self.structure_branch_separate_classifier:
                self.cls_emb_s = nn.Parameter(torch.randn(num_classes, d_model))
        # auxiliary loss for classification task
        if self.use_auxiliary_prior_loss:
            self.aux_gamma_c = nn.Parameter(torch.ones(n_layers - 1))
            self.aux_beta_c = nn.Parameter(torch.zeros(n_layers - 1))
            if not self.use_linear_classifier:
                self.aux_proj_feat_c = nn.Parameter(
                    self.scale * torch.randn(n_layers - 1, d_model, d_model)
                )
                self.aux_proj_cls_c = nn.Parameter(
                    self.scale * torch.randn(n_layers - 1, d_model, d_model)
                )

    def init_weights(self):
        self.apply(init_weights)
        if not self.use_linear_classifier:
            trunc_normal_(self.cls_emb_c, std=0.02)
            if self.structure_branch_use_prior_loss:
                trunc_normal_(self.cls_emb_s, std=0.02)

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

        self.cls_name = cls_name
        if self.all_cls is not None:
            self.cls_index = [self.all_cls.index(name) for name in cls_name]
        else:
            self.cls_index = list(range(len(cls_name)))

    def forward(self, x, img_metas, img_labels=None):
        x = self._transform_inputs(x)
        B, C, H, W = x.size()
        x = x.view(B, C, -1).permute(0, 2, 1)
        # fc = fs = self.proj_dec(x)
        fc = fs = x
        if self.structure_branch_detach:
            fs = fs.detach()
        aux_fc_list = []
        for blk in self.blocks:
            fc, fs = blk(fc, fs)
            aux_fc_list.append(fc)
        fc = self.norm_c(fc)
        fs = self.norm_s(fs)
        # generate classification masks
        if self.use_linear_classifier:
            masks_c = self.cls_emb_c(fc)
            masks_c = masks_c[:, :, self.cls_index]
        else:
            cls_emb_c = self.cls_emb_c[self.cls_index]
            cls_emb_c = cls_emb_c.expand(x.size(0), -1, -1)
            fc = fc @ self.proj_feat_c
            cls_emb_c = cls_emb_c @ self.proj_cls_c
            fc = fc / fc.norm(dim=-1, keepdim=True)
            cls_emb_c = cls_emb_c / cls_emb_c.norm(dim=-1, keepdim=True)
            masks_c = fc @ cls_emb_c.transpose(1, 2)  # / self.temperature
        masks_c = (
            (masks_c - torch.mean(masks_c, dim=-1, keepdim=True))
            / torch.sqrt(
                torch.var(masks_c, dim=-1, keepdim=True, unbiased=False) + 1e-5
            )
        ) * self.gamma_c + self.beta_c
        B, HW, N = masks_c.size()
        masks_c = masks_c.view(B, H, W, N).permute(0, 3, 1, 2)
        embeds = fs.clone().view(B, H, W, -1).permute(0, 3, 1, 2)
        # auxiliary loss for classification task
        aux_masks_c_list = []
        if self.use_auxiliary_prior_loss:
            for lid, aux_fc in enumerate(aux_fc_list[:-1]):
                if self.use_linear_classifier:
                    aux_masks_c = self.cls_emb_c(aux_fc)
                    aux_masks_c = aux_masks_c[:, :, self.cls_index]
                else:
                    aux_cls_emb_c = self.cls_emb_c[self.cls_index]
                    aux_cls_emb_c = aux_cls_emb_c.expand(x.size(0), -1, -1)
                    aux_fc = aux_fc @ self.aux_proj_feat_c[lid]
                    aux_cls_emb_c = aux_cls_emb_c @ self.aux_proj_cls_c[lid]
                    aux_fc = aux_fc / aux_fc.norm(dim=-1, keepdim=True)
                    aux_cls_emb_c = aux_cls_emb_c / aux_cls_emb_c.norm(
                        dim=-1, keepdim=True
                    )
                    aux_masks_c = aux_fc @ aux_cls_emb_c.transpose(
                        1, 2
                    )  # / self.temperature
                # if self.training:
                aux_masks_c = (
                    (aux_masks_c - torch.mean(aux_masks_c, dim=-1, keepdim=True))
                    / torch.sqrt(
                        torch.var(aux_masks_c, dim=-1, keepdim=True, unbiased=False)
                        + 1e-5
                    )
                ) * self.aux_gamma_c[lid] + self.aux_beta_c[lid]
                aux_masks_c = aux_masks_c.view(B, H, W, N).permute(0, 3, 1, 2)
                aux_masks_c_list.append(aux_masks_c)

        # generate masks by structuring feature
        masks_s = None
        if self.structure_branch_use_prior_loss:
            # if not self.structure_branch_separate_classifier:
            # self.cls_emb_s = self.cls_emb_c
            if self.use_linear_classifier:
                masks_s = self.cls_emb_s(fs)
                masks_s = masks_s[:, :, self.cls_index]
            else:
                cls_emb_s = self.cls_emb_s[self.cls_index]
                cls_emb_s = cls_emb_s.expand(x.size(0), -1, -1)
                fs = fs @ self.proj_feat_s
                cls_emb_s = cls_emb_s @ self.proj_cls_s
                fs = fs / fs.norm(dim=-1, keepdim=True)
                cls_emb_s = cls_emb_s / cls_emb_s.norm(dim=-1, keepdim=True)
                masks_s = fs @ cls_emb_s.transpose(1, 2)  # / self.temperature
            # if self.training:
            masks_s = (
                (masks_s - torch.mean(masks_s, dim=-1, keepdim=True))
                / torch.sqrt(
                    torch.var(masks_s, dim=-1, keepdim=True, unbiased=False) + 1e-5
                )
            ) * self.gamma_s + self.beta_s
            masks_s = masks_s.view(B, H, W, N).permute(0, 3, 1, 2)

        return masks_c, masks_s, embeds, aux_masks_c_list

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        self._update(training=True)
        masks_c, masks_s, embeds, aux_masks_c_list = self.forward(inputs, img_metas)
        losses = self.losses(
            masks_c, masks_s, embeds, aux_masks_c_list, gt_semantic_seg
        )
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, gt_semantic_seg=None, img=None):
        self._update(training=False)
        masks_c, masks_s, embeds, _ = self.forward(inputs, img_metas)
        masks = masks_s if self.structure_inference else masks_c
        if self.oracle_inference:
            assert gt_semantic_seg is not None
            masks = self.oracle_propagation(embeds, gt_semantic_seg)
        if self.visualize_seed:
            if "n02779435_19523" in img_metas[0]["ori_filename"]:
                self.visualize_imagenet(img, masks, embeds, gt_semantic_seg, img_metas)
        return masks

    def visualize_imagenet(self, img, mask, embed, label, img_metas):
        h = label.shape[-2]
        w = label.shape[-1]
        mask = resize(
            mask, size=(h, w), mode="bilinear", align_corners=self.align_corners
        )
        embed = resize(
            embed, size=(h, w), mode="bilinear", align_corners=self.align_corners
        )
        B, N, H, W = mask.shape
        mask = mask.reshape(N, H * W).permute(1, 0)
        label = label.reshape(H * W)
        unique_label = torch.unique(label)
        unique_label = unique_label[unique_label != self.ignore_index]
        assert len(unique_label) == 1
        l = int(unique_label)
        png_name = (
            self.cls_name[l]
            + "_"
            + img_metas[0]["ori_filename"].split("/")[-1].replace("JPEG", "png")
        )
        inds = (mask[:, l] > self.weakly_prior_thresh).nonzero(as_tuple=False).flatten()
        if inds.numel() < self.weakly_min_kept:
            inds = mask[:, l].topk(self.weakly_min_kept).indices
        elif inds.numel() > self.weakly_max_kept:
            inds = mask[:, l].topk(self.weakly_max_kept).indices
        mask = torch.zeros_like(mask[:, l])
        mask[inds] = 1
        embed = embed / embed.norm(dim=-1, keepdim=True)
        B, C, H, W = embed.shape
        embed = embed.reshape(C, H * W).permute(1, 0)
        img = img[0]
        for i in range(3):
            img[i] = (img[i] - img[i].min()) / (img[i].max() - img[i].min())
        # (H, W, 3)
        img = img.permute(1, 2, 0)
        for k in range(10):
            idx = int(np.random.choice(range(len(inds)), size=1))
            cos_map = embed[int(inds[idx])] @ embed.T
            cos_map = (cos_map - cos_map.min()) / (cos_map.max() - cos_map.min())
            cos_map = cos_map.reshape(H, W)
            cos_map_img = torch.stack(
                [cos_map, torch.zeros_like(cos_map) * 0.5, 1 - cos_map], dim=-1
            )
            img_ = img * 0.0 + cos_map_img * 1.0
            h = int(inds[idx]) // H
            w = int(inds[idx]) % H
            img_[h - 2 : h + 2, w - 2 : w + 2, 0] = 1.0
            img_[h - 2 : h + 2, w - 2 : w + 2, 1] = 1.0
            img_[h - 2 : h + 2, w - 2 : w + 2, 2] = 1.0
            img_ = img_.cpu().numpy()
            img_ = (img_ * 255).astype(np.uint8)
            Image.fromarray(img_).save(
                os.path.join(f"structure_{k}_" + png_name), format="PNG"
            )

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

    @force_fp32(apply_to=("seg_mask", "seg_mask_s"))
    def losses(self, seg_mask, seg_mask_s, seg_embed, aux_seg_mask_list, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        h = seg_label.shape[-2] // self.downsample_rate
        w = seg_label.shape[-1] // self.downsample_rate
        # (256, 256), (160, 160)
        seg_mask = resize(
            seg_mask, size=(h, w), mode="bilinear", align_corners=self.align_corners
        )
        seg_embed = resize(
            seg_embed, size=(h, w), mode="bilinear", align_corners=self.align_corners
        )
        seg_label = resize(seg_label.float(), size=(h, w), mode="nearest").long()

        ########## classification task ##########
        B, N, H, W = seg_mask.shape
        if self.dataset_on_gpu in self.weakly_supervised_datasets:
            prior_mask, prior_label, prior_inds = self._weak_sample(seg_mask, seg_label)
        else:
            prior_mask, prior_label, prior_inds = self._sample(seg_mask, seg_label)
        # no valid label
        if prior_mask is None:
            loss["loss_prior"] = seg_mask.sum() * 0.0
        else:
            assert prior_label is not None
            loss["loss_prior"] = (
                self.loss_decode(
                    prior_mask, prior_label, weight=None, ignore_index=self.ignore_index
                )
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
            if self.dataset_on_gpu in self.weakly_supervised_datasets:
                prior_mask, prior_label, _ = self._weak_sample(seg_mask_s, seg_label)
            else:
                prior_mask, prior_label, _ = self._sample(seg_mask_s, seg_label)
            # no valid label
            if prior_mask is None:
                loss["loss_prior_s"] = seg_mask_s.sum() * 0.0
            else:
                assert prior_label is not None
                loss["loss_prior_s"] = (
                    self.loss_decode(
                        prior_mask,
                        prior_label,
                        weight=None,
                        ignore_index=self.ignore_index,
                    )
                    * self.prior_loss_weight
                )
        # auxiliary loss for classification task
        if self.use_auxiliary_prior_loss:
            for lid, aux_seg_mask in enumerate(aux_seg_mask_list):
                aux_seg_mask = resize(
                    aux_seg_mask,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                if self.dataset_on_gpu in self.weakly_supervised_datasets:
                    prior_mask, prior_label, _ = self._weak_sample(
                        aux_seg_mask, seg_label
                    )
                else:
                    prior_mask, prior_label, _ = self._sample(aux_seg_mask, seg_label)
                # no valid label
                if prior_mask is None:
                    loss[f"loss_prior_{lid}"] = aux_seg_mask.sum() * 0.0
                else:
                    assert prior_label is not None
                    loss[f"loss_prior_{lid}"] = (
                        self.loss_decode(
                            prior_mask,
                            prior_label,
                            weight=None,
                            ignore_index=self.ignore_index,
                        )
                        * self.prior_loss_weight
                    )
        # log accuracy
        seg_label = seg_label.flatten()
        seg_mask = seg_mask.permute(0, 2, 3, 1).reshape(B * H * W, N)
        imagenet_in_batch = any(["in" in x for x in self.mix_batch_datasets])
        acc_weight = 0.0 if "in" in self.dataset_on_gpu else 2.0
        acc_weight = acc_weight if imagenet_in_batch else 1.0
        loss["acc_seg"] = accuracy(seg_mask, seg_label)

        ########## structuring task ##########
        loss_structure = (
            self.loss_structure(seg_embed, seg_label, prior_inds)
            * self.structure_loss_weight
        )
        if self.dataset_on_gpu in self.weakly_supervised_datasets:
            loss["loss_structure"] = loss_structure * 0.0
            loss["loss_structure_weak"] = loss_structure * 2.0
        else:
            loss["loss_structure"] = loss_structure * 2.0
            loss["loss_structure_weak"] = loss_structure * 0.0
        return loss

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
        unique_label = torch.unique(seg_label)
        unique_label = unique_label[unique_label != self.ignore_index]
        pos_bucket = []
        for l in unique_label:
            pos_bucket.append((seg_label == l).nonzero(as_tuple=False)[:, 0])
        if len(pos_bucket) == 0:
            return None, None, None
        num_per_bucket = []
        for p in pos_bucket:
            k = int(len(p) * self.prior_rate)
            k = max(min_kept, k)
            num_per_bucket.append(k)
        prior_bucket = []
        for k, p, l in zip(num_per_bucket, pos_bucket, unique_label):
            inds = seg_mask[p, int(l)].topk(k).indices
            prior_bucket.append(p[inds])
        # don't know what happened to cause this
        if len(prior_bucket) == 0:
            return None, None, None
        prior_inds = torch.cat(prior_bucket)
        seg_label = seg_label.reshape(B * H * W)
        prior_mask = seg_mask[prior_inds]
        prior_label = seg_label[prior_inds]
        return prior_mask, prior_label, prior_inds

    def _weak_sample(self, seg_mask, seg_label):
        """
        for each image, smaple each category separately
        i.e. a pixel could be assigned to more than one categories
        """
        B, N, H, W = seg_mask.size()
        prior_mask, prior_label, prior_inds = [], [], []
        for mask, label in zip(seg_mask, seg_label):
            mask = mask.reshape(N, H * W).permute(1, 0)
            label = label.reshape(H * W)
            unique_label = torch.unique(label)
            unique_label = unique_label[unique_label != self.ignore_index]
            prior_ind = []
            for l in unique_label:
                inds = (
                    (mask[:, l] > self.weakly_prior_thresh)
                    .nonzero(as_tuple=False)
                    .flatten()
                )
                if inds.numel() < self.weakly_min_kept:
                    inds = mask[:, l].topk(self.weakly_min_kept).indices
                elif inds.numel() > self.weakly_max_kept:
                    inds = mask[:, l].topk(self.weakly_max_kept).indices
                prior_mask.append(mask[inds])
                prior_label.append(torch.ones_like(label[inds]) * l)
                prior_ind.append(inds)
            prior_inds.append(prior_ind)
        prior_mask = torch.cat(prior_mask, dim=0)
        prior_label = torch.cat(prior_label, dim=0)
        return prior_mask, prior_label, prior_inds

    def loss_structure(self, seg_embed, seg_label, prior_inds):
        # turn off weakly structure loss
        if self.structure_loss_weight == 0.0:
            return seg_embed.sum() * 0.0
        B, C, H, W = seg_embed.size()
        if self.dataset_on_gpu in self.weakly_supervised_datasets:
            assert isinstance(prior_inds, list) and len(prior_inds) == B
            seg_embed = seg_embed.permute(0, 2, 3, 1).reshape(B, H * W, C)
            seg_label = seg_label.reshape(B, H * W)
            loss, num_devide = 0.0, 0
            for feat, label, prior_ind in zip(seg_embed, seg_label, prior_inds):
                pos_bucket = [torch.arange(len(label))]
                pos_inds = self._sample_random_points(pos_bucket)
                sample_feat = torch.cat([feat[i] for i in pos_inds], dim=0)
                for ind in prior_ind:
                    prior_feat = feat[ind]
                    loss += self.loss_similarity(sample_feat, None, prior_feat)
                    num_devide += 1
            loss = loss / num_devide
        else:
            seg_embed = seg_embed.permute(0, 2, 3, 1).reshape(B * H * W, C)
            seg_label = seg_label.reshape(B * H * W)
            unique_label = torch.unique(seg_label)
            pos_bucket = [
                torch.nonzero(seg_label == l)[:, 0]
                for l in unique_label
                if l != self.ignore_index
            ]
            if len(pos_bucket) == 0:
                return seg_embed[seg_label != self.ignore_index].sum()
            pos_inds = self._sample_random_points(pos_bucket)
            sample_cls = torch.cat([seg_label[[i]] for i in pos_inds], dim=0).to(
                seg_embed.device
            )
            sample_feat = torch.cat([seg_embed[i] for i in pos_inds], dim=0)
            loss = self.loss_similarity(sample_feat, sample_cls)
        return loss

    def loss_similarity(self, feat, label, prior_feat=None):
        """Compute the similarity loss
        Args:
            feat (torch.Tensor): [N, C]
            label (torch.Tensor): [N]
            prior_feat (torch.Tensor): [Np, C]
        """
        feat = feat / feat.norm(dim=-1, keepdim=True)
        if self.dataset_on_gpu in self.weakly_supervised_datasets:
            assert prior_feat is not None
            prior_feat = prior_feat / prior_feat.norm(dim=-1, keepdim=True)
            prior_cos_sim = feat @ prior_feat.T
            pos_inds = (
                prior_cos_sim.max(dim=-1).values > self.weakly_structure_pos_thresh
            )
            if pos_inds.sum() == 0:
                pos_inds = (
                    prior_cos_sim.max(dim=-1)
                    .values.topk(self.weakly_pos_sample_num)
                    .indices
                )
            feat = feat[pos_inds]
            cos_sim = feat @ feat.T
            label_sim = torch.ones_like(cos_sim)
        else:
            assert label is not None
            cos_sim = feat @ feat.T
            label_sim = (label[None, :] == label[:, None]).int().float()
        valid_mask = torch.ones_like(cos_sim)
        valid_mask[
            torch.arange(len(valid_mask)).to(valid_mask.device),
            torch.arange(len(valid_mask)).to(valid_mask.device),
        ] = 0
        cos_sim = cos_sim[valid_mask.bool()]
        label_sim = label_sim[valid_mask.bool()]
        # NOTE: for negative samples, don't add loss if they are lower than the thresh
        _mask = (cos_sim > self.structure_loss_thresh) | (label_sim == 1)
        if _mask.sum() == 0:
            return cos_sim.sum() * 0.0
        cos_sim = cos_sim[_mask]
        label_sim = label_sim[_mask]
        return torch.pow(cos_sim - label_sim, 2).mean()

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

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PropagateAttention(nn.Module):
    def __init__(self, dim, heads, dropout, gamma_init, use_baseline):
        super().__init__()
        self.heads = heads
        self.use_baseline = use_baseline
        head_dim = dim // heads
        self.scale = head_dim**-0.5
        # gamma is a learnable scalar for each head
        if not self.use_baseline:
            self.gamma = nn.Parameter(torch.ones(heads) * gamma_init)

        self.qc_linear = nn.Linear(dim, dim, bias=False)
        self.qs_linear = nn.Linear(dim, dim, bias=False)
        self.kc_linear = nn.Linear(dim, dim, bias=False)
        self.vc_linear = nn.Linear(dim, dim, bias=False)
        self.vs_linear = nn.Linear(dim, dim, bias=False)
        if self.use_baseline:
            self.ks_linear = nn.Linear(dim, dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_c = nn.Linear(dim, dim)
        self.proj_s = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, fc, fs):
        assert fc.shape == fs.shape, f"{fc.shape} != {fs.shape}"
        B, _, C = fc.shape
        # B, head, N, C // head
        qc = (
            self.qc_linear(fc)
            .reshape(B, -1, self.heads, C // self.heads)
            .permute(0, 2, 1, 3)
        )
        qs = (
            self.qs_linear(fs)
            .reshape(B, -1, self.heads, C // self.heads)
            .permute(0, 2, 1, 3)
        )
        kc = (
            self.kc_linear(fc)
            .reshape(B, -1, self.heads, C // self.heads)
            .permute(0, 2, 1, 3)
        )
        vc = (
            self.vc_linear(fc)
            .reshape(B, -1, self.heads, C // self.heads)
            .permute(0, 2, 1, 3)
        )
        vs = (
            self.vs_linear(fs)
            .reshape(B, -1, self.heads, C // self.heads)
            .permute(0, 2, 1, 3)
        )

        attn_c = (qc @ kc.transpose(-2, -1)) * self.scale
        if self.use_baseline:
            ks = (
                self.ks_linear(fs)
                .reshape(B, -1, self.heads, C // self.heads)
                .permute(0, 2, 1, 3)
            )
            attn_s = (qs @ ks.transpose(-2, -1)) * self.scale
        else:
            qs = qs / qs.norm(dim=-1, keepdim=True)
            attn_s = (qs @ qs.transpose(-2, -1)) * self.gamma[None, :, None, None]
        attn1 = (attn_c + attn_s.detach()).softmax(dim=-1)
        attn2 = (attn_c.detach() + attn_s).softmax(dim=-1)
        # attn = self.attn_drop(attn)
        attn1 = self.attn_drop(attn1)
        attn2 = self.attn_drop(attn2)

        fc = (attn1 @ vc).transpose(1, 2).reshape(B, -1, C)
        fs = (attn2 @ vs).transpose(1, 2).reshape(B, -1, C)
        fc = self.proj_c(fc)
        fs = self.proj_s(fs)
        fc = self.proj_drop(fc)
        fs = self.proj_drop(fs)
        return fc, fs


class Block(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        mlp_dim,
        dropout,
        drop_path,
        gamma_init=1.0,
        use_baseline=False,
    ):
        super().__init__()
        self.normc1 = nn.LayerNorm(dim)
        self.norms1 = nn.LayerNorm(dim)
        self.normc2 = nn.LayerNorm(dim)
        self.norms2 = nn.LayerNorm(dim)
        self.attn = PropagateAttention(dim, heads, dropout, gamma_init, use_baseline)
        self.mlp_c = FeedForward(dim, mlp_dim, dropout)
        self.mlp_s = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, fc, fs):
        fc1, fs1 = self.attn(self.normc1(fc), self.norms1(fs))

        fc = fc + self.drop_path(fc1)
        fc1 = self.mlp_c(self.normc2(fc))
        fc = fc + self.drop_path(fc1)

        fs = fs + self.drop_path(fs1)
        fs1 = self.mlp_s(self.norms2(fs))
        fs = fs + self.drop_path(fs1)
        return fc, fs
