# Copyright (c) OpenMMLab. All rights reserved.
import torch
from genericpath import exists
from .builder import DATASETS
from .custom import CustomDataset

import random
import os.path as osp
from PIL import Image
import numpy as np
import mmcv
from mmcv.utils import print_log
from mmcv.parallel import DataContainer as DC
from mmseg.utils import get_root_logger


@DATASETS.register_module()
class ImageNet21K(CustomDataset):

    # define CLASSES, PALETTE here to satisfy mmseg assertions
    CLASSES = ('bookcase',)

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


@DATASETS.register_module()
class ImageNet130(ImageNet21K):

    CLASSES = (
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 
        'bed', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 
        'water', 'sofa', 'shelf', 'house', 'mirror', 'rug', 'field', 
        'armchair', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 
        'railing', 'cushion', 'base', 'box', 'column', 'chest of drawers', 'counter', 
        'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'stairs', 'runway', 
        'case', 'pool table', 'screen door', 'river', 'bridge', 'bookcase', 'blind', 
        'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'stove', 
        'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 
        'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 
        'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 
        'pole', 'bannister', 'ottoman', 'bottle', 'buffet', 'stage', 'van', 
        'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'stool', 'basket', 
        'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 
        'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 
        'dishwasher', 'blanket', 'sculpture', 'sconce', 'vase', 'traffic light', 'tray', 
        'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'shower', 
        'radiator', 'glass', 'clock', 'flag')

    IMAGE_IDS = (
        "n04547592", "n02913152", "n09436708", "n03365592", "n13104059", 
        "n02990373", "n04096066", "n02818832", "n04587648", "n08598301", 
        "n02933112", "n04215402", "n00007846", "n03462747", "n03221720", 
        "n04379243", "n09359803", "n00017222", "n03151077", "n03002096", 
        "n02958343", "n07935504", "n04256520", "n04190052", "n03544360", 
        "n03773035", "n04118021", "n08659446", "n02738535", "n03327234", 
        "n03179701", "n09416076", "n04550184", "n03636248", "n02808440", 
        "n04047401", "n03938244", "n02797692", "n02883344", "n03073977", 
        "n03015254", "n03116530", "n15019030", "n02998563", "n04233124", 
        "n03346455", "n04070727", "n04314914", "n04120842", "n02975212", 
        "n03982430", "n04153025", "n09415584", "n02898711", "n02870880", 
        "n02851099", "n03063968", "n04446276", "n11669921", "n02870526", 
        "n09303008", "n02828884", "n04330267", "n12582231", "n03619890", 
        "n03082979", "n04373704", "n02858304", "n02789487", "n02706806", 
        "n03547054", "n02924116", "n04459362", "n03665366", "n04490091", 
        "n04460130", "n03005285", "n02763901", "n04335886", "n02874086", 
        "n04405907", "n02691156", "n14844693", "n02728440", "n03976657", 
        "n02788148", "n03858418", "n02876657", "n02912065", "n04296562", 
        "n04520170", "n04194289", "n03388043", "n03100897", "n02951843", 
        "n04554684", "n04326896", "n02801938", "n04411264", "n02774152", 
        "n03790512", "n03125729", "n03862676", "n02779435", "n00021265", 
        "n04388743", "n04217882", "n03761084", "n03991062", "n00015388", 
        "n02834778", "n09332890", "n03207941", "n02849154", "n04157320", 
        "n04148703", "n04522168", "n06874185", "n04476259", "n02747177", 
        "n03320046", "n03934042", "n03085602", "n03960490", "n03782190", 
        "n04208936", "n04041069", "n03438661", "n03046257", "n03354903")

    PALETTE = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        results = self.pipeline(results)
        # prepare
        filename = img_info["filename"]
        if "/" in filename:
            img_id = filename.split("/")[0]
        else:
            img_id = filename.rstrip(self.img_suffix)
        try:
            _class = self.IMAGE_IDS.index(img_id)
        except:
            assert False, f"{img_id}"
        results["gt_semantic_seg"] = DC(
                torch.zeros_like(results['gt_semantic_seg'].data) + _class
        , stack=True)
        return results


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """
    from collections.abc import Sequence
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')