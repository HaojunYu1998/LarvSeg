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
    CLASSES = ('person', )

    PALETTE = [[0, 192, 64],]

    def __init__(self, **kwargs):
        super(ImageNet21K, self).__init__(img_suffix=".jpg", seg_map_suffix=".png", **kwargs)
        # load annotations
    #     self.img_infos = self.load_annotations(
    #         self.img_dir, self.img_suffix, self.ann_dir, self.seg_map_suffix, self.split
    #     )
    
    # def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
    #     """Load annotation from directory.

    #     Args:
    #         img_dir (str): Path to image directory
    #         img_suffix (str): Suffix of images.
    #         ann_dir (str|None): Path to annotation directory.
    #         seg_map_suffix (str|None): Suffix of segmentation maps.
    #         split (str|None): Split txt file. If split is specified, only file
    #             with suffix in the splits will be loaded. Otherwise, all images
    #             in img_dir/ann_dir will be loaded. Default: None

    #     Returns:
    #         list[dict]: All image info of dataset.
    #     """

    #     img_infos = []
    #     if split is not None:
    #         with open(split) as f:
    #             for line in f:
    #                 img_name = line.strip()
    #                 img_info = dict(filename=img_name + img_suffix)
    #                 if ann_dir is not None:
    #                     seg_map = img_name + seg_map_suffix
    #                     img_info["ann"] = dict(seg_map=seg_map)
    #                 img_infos.append(img_info)
    #     else:
    #         for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
    #             seg_map = img.replace(img_suffix, seg_map_suffix)
    #             if not osp.exists(osp.join(ann_dir, seg_map)):
    #                 print(f"{osp.join(ann_dir, seg_map)} NOT EXISTS!")
    #                 continue
    #             img_info = dict(filename=img)
    #             if ann_dir is not None:
    #                 img_info["ann"] = dict(seg_map=seg_map)
    #             img_infos.append(img_info)
    #         img_infos = sorted(img_infos, key=lambda x: x["filename"])

    #     # print_log(f"Loaded {len(img_infos)} images", logger=get_root_logger())
    #     return img_infos

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