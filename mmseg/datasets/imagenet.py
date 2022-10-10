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
        try:
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
            _class = self.IMAGE_IDS.index(img_id)
            results["gt_semantic_seg"] = DC(
                torch.zeros_like(results['gt_semantic_seg'].data) + _class
            , stack=True)
        except:
            for _ in range(1000):
                try:
                    import numpy as np
                    idx_ = int(np.random.uniform() * (len(self.img_infos)-1))
                    img_info = self.img_infos[idx_]
                    ann_info = self.get_ann_info(idx_)
                    results = dict(img_info=img_info, ann_info=ann_info)
                    self.pre_pipeline(results)
                    results = self.pipeline(results)
                    # prepare
                    filename = img_info["filename"]
                    if "/" in filename:
                        img_id = filename.split("/")[0]
                    else:
                        img_id = filename.rstrip(self.img_suffix)
                    _class = self.IMAGE_IDS.index(img_id)
                    results["gt_semantic_seg"] = DC(
                        torch.zeros_like(results['gt_semantic_seg'].data) + _class
                    , stack=True)
                    print(f"{idx} is invalid, sample {idx_} instead")
                    break
                except:
                    continue
        return results


@DATASETS.register_module()
class ImageNet585(ImageNet21K):

    CLASSES = (
        'aircraft-carrier', 'stage', 'key', 'vent', 'alga', 'ashcan', 'tunnel', 
        'drawer', 'throne', 'workbench', 'swimming-pool', 'computer', 'buffet', 'switch', 
        'rake', 'funnel', 'plant', 'tongs', 'oven', 'stall', 'pantry', 
        'food', 'easel', 'lectern', 'viaduct', 'scoreboard', 'videocassette-recorder', 'wardrobe', 
        'sweatshirt', 'map', 'vase', 'air-conditioner', 'calculator', 'tower', 'control-tower', 
        'basket', 'arch', 'mast', 'skirt', 'grass', 'push-button', 'sunglasses', 
        'organ', 'support', 'washer', 'office', 'windmill', 'shower-stall', 'flip-flop', 
        'console-table', 'tree', 'shower', 'beacon', 'plaything', 'ball', 'field', 
        'spray', 'spanners', 'telescope', 'scaffolding', 'car', 'umbrella', 'monument', 
        'gym-shoe', 'soap-dish', 'ski-slope', 'synthesizer', 'chest-of-drawers', 'sword', 'person', 
        'funfair', 'skyscraper', 'double-door', 'helicopter', 'bag', 'folders', 'net', 
        'shoe', 'frame', 'postbox', 'dumbbells', 'bus', 'soap-dispenser', 'ashtray', 
        'coffee-maker', 'conveyer-belt', 'root', 'dome', 'golf-bag', 'watering-can', 'shutter', 
        'roof-rack', 'stool', 'sport-basket', 'cat', 'mouse', 'games', 'music-stand', 
        'patio', 'tractor', 'signboard', 'alarm-clock', 'ladle', 'tureen', 'mirror', 
        'bowl', 'bidet', 'wire', 'food-processor', 'cellular-telephone', 'canoe', 'fireplace', 
        'chips', 'machine', 'boot', 'pumpkin', 'valley', 'snow', 'bat', 
        'wheel', 'sea', 'candlestick', 'belt', 'table-cloth', 'cooker', 'laptop', 
        'casserole', 'roll', 'hammock', 'trough', 'elevator-door', 'document', 'monitor', 
        'notebook', 'dirt-track', 'cue', 'tramway', 'boat', 'column', 'lock', 
        'teapot', 'plate', 'statue', 'bell', 'crt-screen', 'parterre', 'water-faucet', 
        'desk', 'fan', 'wallet', 'knife', 'sofa', 'covered-bridge', 'rock', 
        'dummy', 'bird', 'gear', 'mug', 'towel', 'radiator', 'crane', 
        'box', 'door', 'dollhouse', 'ice-floe', 'railway', 'blackboard', 'windowpane', 
        'sconce', 'pier', 'streetlight', 'saucepan', 'banner', 'trouser', 'curb', 
        'dashboard', 'hen', 'canister', 'iceberg', 'patty', 'shop', 'garage', 
        'cutlery', 'weighbridge', 'pallet', 'cabin', 'scale', 'streetcar', 'airplane', 
        'butane-gas-cylinder', 'roulette', 'truck', 'boiler', 'heater', 'table', 'toilet', 
        'goal', 'water-mill', 'bridge', 'carapace', 'house', 'cockpit', 'scarf', 
        'rocking-chair', 'stapler', 'counter', 'big-top', 'street-number', 'aquarium', 'tie', 
        'tins', 'hotplate', 'glacier', 'labyrinth', 'flag', 'weeds', 'wall', 
        'van', 'greenhouse', 'eaves', 'altarpiece', 'folding-screen', 'sea-star', 'awning', 
        'hood', 'kitchen-island', 'sprinkler', 'service-elevator', 'grandstand', 'terminal', 'necklace', 
        'grille-door', 'paper-towel', 'treadmill', 'backpack', 'plaque', 'sugar-bowl', 'steam-shovel', 
        'keyboard', 'caravan', 'trestle', 'rug', 'dishwasher', 'stairs', 'trench', 
        'shipyard', 'kennel', 'candelabrum', 'guitar', 'hot-tub', 'dvds', 'blanket', 
        'pen', 'canyon', 'chandelier', 'mitten', 'hard-drive', 'pottery', 'swivel-chair', 
        'target', 'sand-trap', 'ice', 'coffee-table', 'lamp', 'tent', 'building', 
        'shovel', 'hole', 'lab-bench', 'side', 'golf-cart', 'ironing-board', 'fence', 
        'train', 'portable-fridge', 'seat', 'canopy', 'playground', 'shelf', 'hair-spray', 
        'parking-meter', 'structure', 'court', 'alarm', 'racket', 'radio', 'magazine', 
        'newspaper', 'horse', 'motorboat', 'coffin', 'mask', 'booth', 'cushion', 
        'grating', 'wheelbarrow', 'pergola', 'sleeping-robe', 'smoothing-iron', 'fire-extinguisher', 'gravestone', 
        'telephone', 'bowling-alley', 'container', 'case', 'apron', 'dam', 'hair-dryer', 
        'telephone-booth', 'skimmer', 'valve', 'bicycle-rack', 'wheelchair', 'curtain', 'track', 
        'helmet', 'tray', 'stick', 'palm', 'finger', 'tape', 'sweater', 
        'toaster', 'hill', 'bench', 'alembic', 'coat', 'screen-door', 'light-bulb', 
        'cross', 'ear', 'bookcase', 'envelopes', 'beam', 'forklift', 'steering-wheel', 
        'vault', 'grand-piano', 'ceiling', 'engine', 'brick', 'jar', 'shaving-brush', 
        'bathtub', 'road', 'gate', 'tarp', 'ruins', 'boxing-ring', 'roller-coaster', 
        'microphone', 'bucket', 'ice-hockey-rink', 'kettle', 'hay', 'fountain', 'well', 
        'microwave', 'place-mat', 'base', 'hovel', 'bottle', 'mill', 'hairbrush', 
        'refrigerator', 'baptismal-font', 'confessional-booth', 'box-office', 'towel-rack', 'drum', 'ski-lift', 
        'leash', 'sand-dune', 'briefcase', 'jack', 'broom', 'safety-belt', 'branch', 
        'forecourt', 'salt', 'pot', 'apparel', 'bannister', 'faucet', 'cup', 
        'watch', 'altar', 'bus-stop', 'tapestry', 'watchtower', 'tripod', 'paper', 
        'tank', 'pool-table', 'eiderdown', 'gear-shift', 'water-wheel', 'jersey', 'shower-room', 
        'sidewalk', 'pepper-shaker', 'cabinet', 'hammer', 'revolving-door', 'water-tower', 'shore', 
        'cds', 'typewriter', 'toilet-tissue', 'porch', 'catwalk', 'spotlight', 'equipment', 
        'minibike', 'coat-rack', 'doorframe', 'bonfire', 'skylight', 'carport', 'flower', 
        'obelisk', 'stove', 'drinking-glass', 'trailer', 'controls', 'barrel', 'onions', 
        'clock', 'remote-control', 'sand', 'dog-dish', 'arcade-machine', 'antenna', 'piano', 
        'billboard', 'sink', 'tomb', 'brush', 'television-camera', 'hat', 'dishrag', 
        'bulldozer', 'grinder', 'trunk', 'henhouse', 'recycling-bin', 'pitcher', 'plate-rack', 
        'spoon', 'locker', 'lockers', 'menu', 'laptop-bag', 'loudspeaker', 'witness-stand', 
        'television-receiver', 'dish-rack', 'traffic-light', 'fish', 'sewing-machine', 'ramp', 'star', 
        'padlock', 'gas-pump', 'sun', 'tallboy', 'teacup', 'cliff', 'blender', 
        'golf-club', 'tools', 'hanger', 'fruit', 'palette', 'cash-register', 'inflatable-glove', 
        'candy', 'niche', 'excavator', 'arcade', 'rope', 'cap', 'shaker', 
        'printer', 'frying-pan', 'bread', 'fire-alarm', 'shrine', 'mezzanine', 'armchair', 
        'bowling-pins', 'embankment', 'blind', 'ship', 'dog', 'shirt', 'traffic-cone', 
        'fork', 'board', 'barrier', 'animal', 'barbecue', 'punching-bag', 'bicycle', 
        'partition', 'rubbish', 'scissors', 'vending-machine', 'andiron', 'magazine-rack', 'stretcher', 
        'file-cabinet', 'sky', 'bleachers', 'chain', 'photocopier', 'bar', 'pool-ball', 
        'balloon', 'tricycle', 'gravy-boat', 'book', 'towel-dispenser', 'hook', 'chair', 
        'light', 'carriage', 'meter', 'railing', 'pipe', 'rifle', 'runway', 
        'earth', 'carousel', 'rod', 'blast-furnace', 'floor', 'windshield', 'spectacles', 
        'water', 'wall-socket', 'slot-machine', 'sheet', 'podium', 'gazebo', 'exhibitor', 
        'barbed-wire', 'candle', 'manhole', 'potatoes', 'can', 'cradle', 'ferris-wheel', 
        'sauna', 'cart', 'temple', 'projector', 'violin', 'water-cooler', 'system', 
        'pole', 'shawl', 'stethoscope', 'pack', 'roof', 'napkin', 'ottoman', 
        'mountain', 'bed', 'shelter', 'trellis', 'fireplace-utensils', 'rocket', 'spice-rack', 
        'deck-chair', 'shopping-carts', 'sculpture', 'baby-buggy', 'reel', 'amphitheater', 'jacket', 
        'sofa-bed', 'climbing-frame', 'towel-rail', 'cannon', )

    IMAGE_IDS = (
        'n02687172', 'n04296562', 'n03613294', 'n04526520', 'n01397114', 'n02747177', 'n09230041', 
        'n03233905', 'n04429376', 'n04600486', 'n00442115', 'n03082979', 'n02912065', 'n04372370', 
        'n04050066', 'n03403643', 'n00017222', 'n04450749', 'n03862676', 'n04299215', 'n03885535', 
        'n00021265', 'n03262809', 'n03653583', 'n04532670', 'n04149813', 'n04533802', 'n04550184', 
        'n04370456', 'n03720163', 'n04522168', 'n02686379', 'n02938886', 'n04460130', 'n03098959', 
        'n02801938', 'n02733524', 'n03726760', 'n04230808', 'n12102133', 'n04027023', 'n04356056', 
        'n03854065', 'n04359589', 'n04554684', 'n03841666', 'n04587559', 'n04209613', 'n04241394', 
        'n03092883', 'n13104059', 'n04208936', 'n02814860', 'n04461879', 'n02778669', 'n08659446', 
        'n02754103', 'n04606574', 'n04403638', 'n04141712', 'n02958343', 'n04507155', 'n03743902', 
        'n03472535', 'n04254009', 'n09436444', 'n04376400', 'n03015254', 'n04373894', 'n00007846', 
        'n08494231', 'n04233124', 'n03226880', 'n03512147', 'n02773037', 'n03376279', 'n03819994', 
        'n13926786', 'n03390983', 'n03710193', 'n03255030', 'n02924116', 'n04254120', 'n02747802', 
        'n03063338', 'n03100897', 'n13125117', 'n03220692', 'n03445617', 'n04560292', 'n04211356', 
        'n03696301', 'n04326896', 'n02802215', 'n02121620', 'n03793489', 'n03413828', 'n03801760', 
        'n03899768', 'n04465666', 'n04217882', 'n02694662', 'n03633091', 'n04499062', 'n03773035', 
        'n02881193', 'n02836174', 'n04594489', 'n03378174', 'n02992529', 'n02951358', 'n03346455', 
        'n03020416', 'n03699975', 'n02873520', 'n07735510', 'n09468604', 'n11508382', 'n03132076', 
        'n04574999', 'n09376198', 'n02948557', 'n02827606', 'n03309808', 'n03101156', 'n03642806', 
        'n02978753', 'n04101375', 'n03482252', 'n04488427', 'n03281145', 'n06470073', 'n03782006', 
        'n03832673', 'n14844693', 'n03145522', 'n04469813', 'n02858304', 'n03073977', 'n03683341', 
        'n04398044', 'n03960490', 'n04306847', 'n02824448', 'n04152593', 'n03417749', 'n04559451', 
        'n03179701', 'n03320046', 'n04548362', 'n03623556', 'n04256520', 'n03122073', 'n09416076', 
        'n02848921', 'n01503061', 'n03431243', 'n03797390', 'n04459362', 'n04040759', 'n03126707', 
        'n02883344', 'n03222318', 'n03219483', 'n09309168', 'n04463679', 'n02846511', 'n04587648', 
        'n04148579', 'n03933933', 'n04335886', 'n04138977', 'n02788021', 'n04488530', 'n03149135', 
        'n03163222', 'n01514859', 'n02949542', 'n09308572', 'n07663899', 'n04202417', 'n03416489', 
        'n03154073', 'n04570958', 'n03879456', 'n02932400', 'n04141838', 'n04335435', 'n02691156', 
        'n03156279', 'n13908580', 'n04490091', 'n02863750', 'n03508101', 'n04379964', 'n04446276', 
        'n03442756', 'n04561422', 'n02898711', 'n09432283', 'n03544360', 'n03061505', 'n04143897', 
        'n04099969', 'n04303497', 'n03116530', 'n03035252', 'n03223553', 'n02732072', 'n03815615', 
        'n04438897', 'n03543254', 'n09289331', 'n03733281', 'n03354903', 'n13085113', 'n14564779', 
        'n04520170', 'n03457902', 'n03263076', 'n02699770', 'n04152387', 'n02317335', 'n02763901', 
        'n03531546', 'n03619890', 'n03180969', 'n03394149', 'n04295881', 'n04413419', 'n03814906', 
        'n02936714', 'n03887697', 'n04477219', 'n02769748', 'n02892201', 'n04350581', 'n04310507', 
        'n03614007', 'n04520382', 'n04479694', 'n04118021', 'n03207941', 'n04314914', 'n04478657', 
        'n03216828', 'n03610524', 'n02947818', 'n03467517', 'n03543603', 'n04533946', 'n02849154', 
        'n03906997', 'n09233446', 'n03005285', 'n03775071', 'n03209666', 'n03992703', 'n04373704', 
        'n04394261', 'n02920369', 'n03558176', 'n03063968', 'n03636248', 'n04411264', 'n02913152', 
        'n04208427', 'n09304750', 'n03630262', 'n09437454', 'n03445924', 'n03585875', 'n03327234', 
        'n04468005', 'n03273913', 'n04161358', 'n02951843', 'n03963645', 'n04190052', 'n03476991', 
        'n03891332', 'n04341686', 'n03120491', 'n02694426', 'n04039381', 'n06277135', 'n06595351', 
        'n03822171', 'n02374451', 'n03790230', 'n03064758', 'n03725035', 'n02874086', 'n03938244', 
        'n03454536', 'n02797295', 'n02732827', 'n04097866', 'n03584829', 'n03345837', 'n03455488', 
        'n04401088', 'n02882190', 'n03094503', 'n02975212', 'n02730930', 'n03160309', 'n03483316', 
        'n04401680', 'n04229959', 'n04519153', 'n02835829', 'n04576002', 'n03151077', 'n00440039', 
        'n03513137', 'n04476259', 'n04317833', 'n12582231', 'n03341153', 'n04392113', 'n04370048', 
        'n04442312', 'n09303008', 'n02828884', 'n02696246', 'n03057021', 'n04153025', 'n03665924', 
        'n03135532', 'n13133613', 'n02870880', 'n13869788', 'n02815950', 'n03384352', 'n04313503', 
        'n04523525', 'n03452741', 'n02990373', 'n03288003', 'n02897820', 'n03593526', 'n04185946', 
        'n02808440', 'n04096066', 'n03427296', 'n04395024', 'n04118635', 'n00445802', 'n04102406', 
        'n03759954', 'n02909870', 'n03557360', 'n03612814', 'n07802026', 'n03388043', 'n04572935', 
        'n03761084', 'n03727837', 'n02797692', 'n03547054', 'n02876657', 'n03316406', 'n03475581', 
        'n04070727', 'n02788572', 'n02874214', 'n02885882', 'n04459773', 'n03249569', 'n04231693', 
        'n03652932', 'n09270735', 'n02900705', 'n03588951', 'n02906734', 'n04125853', 'n13163991', 
        'n03382292', 'n07813107', 'n03991062', 'n02728440', 'n02788148', 'n03325088', 'n03147509', 
        'n04555897', 'n02699494', 'n08517676', 'n04393549', 'n04556948', 'n04485082', 'n06255613', 
        'n04388743', 'n03982430', 'n01896844', 'n03432129', 'n04563413', 'n03595523', 'n04209509', 
        'n04215402', 'n03914438', 'n02933112', 'n03481172', 'n04086446', 'n04562935', 'n09433442', 
        'n03079230', 'n04505036', 'n15075141', 'n03984381', 'n03961939', 'n04286575', 'n03294048', 
        'n03790512', 'n03059103', 'n03222722', 'n03343560', 'n04232800', 'n02968074', 'n11669921', 
        'n03837869', 'n04330267', 'n03438257', 'n04467307', 'n03098140', 'n02795169', 'n07722217', 
        'n03046257', 'n04074963', 'n15019030', 'n07557434', 'n02706806', 'n02715229', 'n03928116', 
        'n02839110', 'n02998563', 'n09238926', 'n02908217', 'n04404997', 'n03497657', 'n03207743', 
        'n02916179', 'n03765561', 'n13163553', 'n03016389', 'n04065789', 'n03950228', 'n03961711', 
        'n04284002', 'n02933462', 'n03683606', 'n07565083', 'n02773838', 'n03691459', 'n04596492', 
        'n04405907', 'n03207630', 'n06874185', 'n02512053', 'n04179913', 'n04051549', 'n13881644', 
        'n03874599', 'n03425413', 'n09450163', 'n03518305', 'n04397452', 'n09246464', 'n02850732', 
        'n03446070', 'n04451818', 'n03490884', 'n13134947', 'n03879705', 'n02977438', 'n02885462', 
        'n07597365', 'n04061969', 'n03996416', 'n02733213', 'n04108268', 'n02955065', 'n04183329', 
        'n04004475', 'n03400231', 'n07679356', 'n03343737', 'n04210390', 'n03758089', 'n02738535', 
        'n02882647', 'n03282060', 'n02851099', 'n04194289', 'n02084071', 'n04197391', 'n13872592', 
        'n13914265', 'n03211616', 'n02796623', 'n00015388', 'n04111531', 'n04023962', 'n02834778', 
        'n03894379', 'n09335809', 'n04148054', 'n04525305', 'n02710044', 'n03704549', 'n04336792', 
        'n03337140', 'n09436708', 'n09859684', 'n02999936', 'n03924679', 'n02789487', 'n03982232', 
        'n02782093', 'n04482393', 'n03456024', 'n02870526', 'n03210683', 'n13869547', 'n03001627', 
        'n03665366', 'n02969010', 'n03753077', 'n04047401', 'n13901321', 'n04090263', 'n04120842', 
        'n03462747', 'n02966193', 'n04100174', 'n02849885', 'n09282208', 'n04590553', 'n04272054', 
        'n07935504', 'n04255163', 'n04243941', 'n04188179', 'n03159640', 'n03430418', 'n04038440', 
        'n02790823', 'n02948072', 'n03717447', 'n07710616', 'n02946921', 'n03125729', 'n03329302', 
        'n04139395', 'n03484083', 'n04407686', 'n04009552', 'n04536866', 'n04559166', 'n04377057', 
        'n03988170', 'n04186455', 'n04317175', 'n03870546', 'n04105068', 'n03188531', 'n03858418', 
        'n09359803', 'n02818832', 'n04192238', 'n04478512', 'n04516672', 'n04099429', 'n04275175', 
        'n03168217', 'n04204347', 'n04157320', 'n02766534', 'n02860415', 'n02705201', 'n03590306', 
        'n03100346', 'n03042697', 'n04459909', 'n02950482', )

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
        try:
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
            _class = self.IMAGE_IDS.index(img_id)
            results["gt_semantic_seg"] = DC(
                torch.zeros_like(results['gt_semantic_seg'].data) + _class
            , stack=True)
        except:
            for _ in range(1000):
                try:
                    import numpy as np
                    idx_ = int(np.random.uniform() * (len(self.img_infos)-1))
                    img_info = self.img_infos[idx_]
                    ann_info = self.get_ann_info(idx_)
                    results = dict(img_info=img_info, ann_info=ann_info)
                    self.pre_pipeline(results)
                    results = self.pipeline(results)
                    # prepare
                    filename = img_info["filename"]
                    if "/" in filename:
                        img_id = filename.split("/")[0]
                    else:
                        img_id = filename.rstrip(self.img_suffix)
                    _class = self.IMAGE_IDS.index(img_id)
                    results["gt_semantic_seg"] = DC(
                            torch.zeros_like(results['gt_semantic_seg'].data) + _class
                    , stack=True)
                    print(f"{idx} is invalid, sample {idx_} instead")
                    break
                except:
                    continue
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