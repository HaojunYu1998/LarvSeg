# Copyright (c) OpenMMLab. All rights reserved.
from genericpath import exists
from .builder import DATASETS
from .custom import CustomDataset

import os
import os.path as osp
from PIL import Image
import numpy as np
import mmcv


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
        super(ImageNet21K, self).__init__(img_suffix=".jpg", seg_map_suffix=".png", **kwargs)

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