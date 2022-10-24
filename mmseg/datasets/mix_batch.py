from ast import Not
import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist


# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmcv.runner import get_dist_info
from prettytable import PrettyTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose, LoadAnnotations
from .coco_stuff import COCOStuffDataset
from .ade import ADE20KDataset, ADE20KFULLDataset
from .imagenet import ImageNet21K, ImageNet124, ImageNet130, ImageNet585


@DATASETS.register_module()
class MixBatchDataset(Dataset):
    def __init__(self, dataset_list):
        def build_dataset(cfg):
            args = cfg.copy()
            type = args.pop("type")
            if type == "COCOStuffDataset":
                return COCOStuffDataset(**args)
            elif type == "ADE20KDataset":
                return ADE20KDataset(**args)
            elif type == "ADE20KFULLDataset":
                return ADE20KFULLDataset(**args)
            elif type == "ImageNet21K":
                return ImageNet21K(**args)
            elif type == "ImageNet124":
                return ImageNet124(**args)
            elif type == "ImageNet130":
                return ImageNet130(**args)
            elif type == "ImageNet585":
                return ImageNet585(**args)
            assert False, f"{type} is not supported"

        self.rank, world_size = get_dist_info()
        assert len(dataset_list) <= world_size

        args = dataset_list[self.rank % len(dataset_list)]
        self.dataset = build_dataset(args)

        self.pipeline = self.dataset.pipeline
        self.img_dir = self.dataset.img_dir
        self.img_suffix = self.dataset.img_suffix
        self.ann_dir = self.dataset.ann_dir
        self.seg_map_suffix = self.dataset.seg_map_suffix
        self.split = self.dataset.split
        self.data_root = self.dataset.data_root
        self.test_mode = False
        self.ignore_index = self.dataset.ignore_index
        self.reduce_zero_label = self.dataset.reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.dataset.CLASSES, self.dataset.PALETTE

        self.gt_seg_map_loader = self.dataset.gt_seg_map_loader
        self.int16 = self.dataset.int16

        # load annotations
        self.img_infos = self.dataset.img_infos

    def __len__(self):
        return len(self.dataset.img_infos)

    def load_annotations(self, **kwargs):
        return self.dataset.load_annotations(**kwargs)

    def get_ann_info(self, idx):
        return self.dataset.img_infos[idx]["ann"]

    def __getitem__(self, idx):
        if self.test_mode:
            return self.dataset.prepare_test_img(idx)
        else:
            return self.dataset.prepare_train_img(idx)

    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        raise NotImplementedError

    def get_gt_seg_map_by_idx(self, index):
        return self.dataset.get_gt_seg_map_by_idx(index)

    def get_gt_seg_maps(self, efficient_test=None):
        """Get ground truth segmentation maps for evaluation."""
        if efficient_test is not None:
            warnings.warn(
                "DeprecationWarning: ``efficient_test`` has been deprecated "
                "since MMSeg v0.16, the ``get_gt_seg_maps()`` is CPU memory "
                "friendly by default. "
            )

        for idx in range(len(self)):
            ann_info = self.dataset.get_ann_info(idx)
            results = dict(ann_info=ann_info)
            self.dataset.pre_pipeline(results)
            self.dataset.gt_seg_map_loader(results)
            yield results["gt_semantic_seg"]

    def pre_eval(self, preds, indices):
        raise NotImplementedError

    def get_classes_and_palette(self, classes=None, palette=None):
        return self.dataset.get_classes_and_palette(classes, palette)

    def get_palette_for_custom_classes(self, class_names, palette=None):
        return self.dataset.get_palette_for_custom_classes(class_names, palette)

    def evaluate(self, results, metric="mIoU", logger=None, gt_seg_maps=None, **kwargs):
        raise NotImplementedError