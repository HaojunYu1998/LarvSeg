# Copyright (c) OpenMMLab. All rights reserved.
from genericpath import exists
from .builder import DATASETS
from .custom import CustomDataset

import random
import os.path as osp
from PIL import Image
import numpy as np
import mmcv
from mmcv.utils import print_log
from mmseg.utils import get_root_logger


@DATASETS.register_module()
class ImageNet21K(CustomDataset):

    # define CLASSES, PALETTE here to satisfy mmseg assertions
    CLASSES = (
        'bookcase', 'stool', 'bed', 'vase', 'dirt track', 'buffet', 
        'stove', 'skyscraper', 'bridge', 'box', 'pole', 'grass', 'tent', 
        'flower', 'stairs', 'chair', 'house', 'bench', 'door', 'cushion', 
        'rock', 'bannister', 'trade name', 'car', 'bicycle', 'wall', 'fan', 
        'truck', 'hill', 'shelf', 'kitchen island', 'conveyer belt', 'wardrobe', 
        'airplane', 'refrigerator', 'tower', 'sky', 'ball', 'tray', 'chandelier', 
        'sidewalk', 'wall', 'sand', 'tank', 'stage', 
        'radiator', 'house', 'toilet', 'monitor', 'road', 'awning', 'pool table', 
        'windowpane', 'booth', 'lake', 'radiator', 'base', 'glass', 'swivel chair', 
        'floor', 'radiator', 'cradle', 'river', 'canopy', 'sink', 'arcade machine', 
        'palm', 'bathtub', 'curtain', 'ship', 'streetlight', 'dishwasher', 'plate', 
        'fireplace', 'mirror', 'plant', 'traffic light', 'lamp', 'cabinet', 'truck', 
        'rug', 'water', 'flag', 'bus', 'bar', 'tent', 'fence', 'computer', 'van', 
        'lake', 'minibike', 'table', 'screen door', 'monitor', 'column', 'sconce', 
        'railing', 'towel', 'stove', 'floor', 'kitchen island', 'oven', 'blind', 
        'case', 'bridge', 'lamp', 'monitor', 'pier', 'mountain', 'crt screen', 'railing', 
        'sculpture', 'clock', 'ashcan', 'bar', 'ceiling', 'hovel', 'wardrobe', 'door', 
        'animal', 'bag', 'wall', 'bed', 'ottoman', 'armchair', 'bag', 'chest of drawers', 
        'pole', 'ottoman', 'tree', 'pot', 'bag', 'chair', 'ball', 'door', 'boat', 
        'counter', 'shower', 'bar', 'coffee table', 'glass', 'column', 'wall', 'washer', 
        'double door', 'bar', 'building', 'microwave', 'table', 'book', 'minibike', 
        'lake', 'fountain', 'television receiver', 'bottle', 'desk', 'person', 'grass', 
        'animal', 'apparel', 'animal', 'field', 'palm', 'bathtub', 'blanket', 'light', 
        'tent', 'basket', 'pier', 'runway', 'food', 'sconce', 'sofa', 'earth')

    PALETTE = None

    def __init__(self, **kwargs):
        img_suffix = ".jpg"
        seg_map_suffix = ".png"
        if "img_suffix" in kwargs:
            img_suffix = kwargs["img_suffix"]
            kwargs.pop("img_suffix")
        if "seg_map_suffix" in kwargs:
            seg_map_suffix = kwargs["seg_map_suffix"]
            kwargs.pop("seg_map_suffix")
        super(ImageNet21K, self).__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    if img_suffix == ".JPEG":
                        img_name_ = img_name[:img_name.find("_")] + "/" + img_name
                    else:
                        img_name_ = img_name
                    img_info = dict(filename=img_name_ + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info["ann"] = dict(seg_map=seg_map)
                    # print(img_info)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info["ann"] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x["filename"])

        print_log(f"Loaded {len(img_infos)} images", logger=get_root_logger())
        return img_infos

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)

        while not osp.exists(osp.join(self.ann_dir, ann_info["seg_map"])):
            idx_ = random.choice(
                [i for i in range(len(self.img_infos))]
            )
            img_info = self.img_infos[idx_]
            ann_info = self.get_ann_info(idx_)
            # print(idx_, osp.join(self.ann_dir, ann_info["seg_map"]))
        
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            # result = result + 1

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for ade20k evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)
        return result_files