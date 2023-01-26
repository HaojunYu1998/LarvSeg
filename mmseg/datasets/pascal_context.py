# Copyright (c) OpenMMLab. All rights reserved.

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PascalContextDataset(CustomDataset):
    """PascalContext dataset.

    In segmentation map annotation for PascalContext, 0 stands for background,
    which is included in 60 categories. ``reduce_zero_label`` is fixed to
    False. The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png'.

    Args:
        split (str): Split txt file for PascalContext.
    """

    CLASSES = ('background', 'aeroplane', 'bag', 'bed', 'bedclothes', 'bench',
               'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus',
               'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth',
               'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
               'floor', 'flower', 'food', 'grass', 'ground', 'horse',
               'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person',
               'plate', 'platform', 'pottedplant', 'road', 'rock', 'sheep',
               'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table',
               'track', 'train', 'tree', 'truck', 'tvmonitor', 'wall', 'water',
               'window', 'wood')

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]]

    def __init__(self, split, **kwargs):
        super(PascalContextDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            split=split,
            reduce_zero_label=False,
            **kwargs)
        # assert self.file_client.exists(self.img_dir) and self.split is not None


@DATASETS.register_module()
class PascalContextDataset59(CustomDataset):
    """PascalContext dataset.

    In segmentation map annotation for PascalContext59, background is not
    included in 59 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed
    to '.png'.

    Args:
        split (str): Split txt file for PascalContext.
    """

    CLASSES = ('aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle',
               'bird', 'boat', 'book', 'bottle', 'building', 'bus', 'cabinet',
               'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow',
               'cup', 'curtain', 'dog', 'door', 'fence', 'floor', 'flower',
               'food', 'grass', 'ground', 'horse', 'keyboard', 'light',
               'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform',
               'pottedplant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk',
               'sign', 'sky', 'snow', 'sofa', 'table', 'track', 'train',
               'tree', 'truck', 'tvmonitor', 'wall', 'water', 'window', 'wood')

    PALETTE = [[180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
               [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230],
               [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61],
               [120, 120, 70], [8, 255, 51], [255, 6, 82], [143, 255, 140],
               [204, 255, 4], [255, 51, 7], [204, 70, 3], [0, 102, 200],
               [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71],
               [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92],
               [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6],
               [10, 255, 71], [255, 41, 10], [7, 255, 255], [224, 255, 8],
               [102, 8, 255], [255, 61, 6], [255, 194, 7], [255, 122, 8],
               [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255],
               [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140],
               [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0],
               [255, 224, 0], [153, 255, 0], [0, 0, 255], [255, 71, 0],
               [0, 235, 255], [0, 173, 255], [31, 0, 255]]

    # def __init__(self, split, **kwargs):
    def __init__(self, **kwargs):
        super(PascalContextDataset59, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            # split=split,
            reduce_zero_label=True,
            **kwargs)
        # assert self.file_client.exists(self.img_dir) and self.split is not None


@DATASETS.register_module()
class PascalContextDataset459(CustomDataset):
    """PascalContext dataset.

    In segmentation map annotation for PascalContext59, background is not
    included in 59 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed
    to '.png'.

    Args:
        split (str): Split txt file for PascalContext.
    """

    CLASSES = ('accordion', 'aeroplane', 'air conditioner', 'antenna', 'artillery', 'ashtray', 'atrium', 
            'baby carriage', 'bag', 'ball', 'balloon', 'bamboo weaving', 'barrel', 'baseball bat', 
            'basket', 'basketball backboard', 'bathtub', 'bed', 'bedclothes', 'beer', 'bell', 
            'bench', 'bicycle', 'binoculars', 'bird', 'bird cage', 'bird feeder', 'bird nest', 
            'blackboard', 'board', 'boat', 'bone', 'book', 'bottle', 'bottle opener', 
            'bowl', 'box', 'bracelet', 'brick', 'bridge', 'broom', 'brush', 
            'bucket', 'building', 'bus', 'cabinet', 'cabinet door', 'cage', 'cake', 
            'calculator', 'calendar', 'camel', 'camera', 'camera lens', 'can', 'candle', 
            'candle holder', 'cap', 'car', 'card', 'cart', 'case', 'casette recorder', 
            'cash register', 'cat', 'cd', 'cd player', 'ceiling', 'cell phone', 'cello', 
            'chain', 'chair', 'chessboard', 'chicken', 'chopstick', 'clip', 'clippers', 
            'clock', 'closet', 'cloth', 'clothes tree', 'coffee', 'coffee machine', 'comb', 
            'computer', 'concrete', 'cone', 'container', 'control booth', 'controller', 'cooker', 
            'copying machine', 'coral', 'cork', 'corkscrew', 'counter', 'court', 'cow', 
            'crabstick', 'crane', 'crate', 'cross', 'crutch', 'cup', 'curtain', 
            'cushion', 'cutting board', 'dais', 'disc', 'disc case', 'dishwasher', 'dock', 
            'dog', 'dolphin', 'door', 'drainer', 'dray', 'drink dispenser', 'drinking machine', 
            'drop', 'drug', 'drum', 'drum kit', 'duck', 'dumbbell', 'earphone', 
            'earrings', 'egg', 'electric fan', 'electric iron', 'electric pot', 'electric saw', 'electronic keyboard', 
            'engine', 'envelope', 'equipment', 'escalator', 'exhibition booth', 'extinguisher', 'eyeglass', 
            'fan', 'faucet', 'fax machine', 'fence', 'ferris wheel', 'fire extinguisher', 'fire hydrant', 
            'fire place', 'fish', 'fish tank', 'fishbowl', 'fishing net', 'fishing pole', 'flag', 
            'flagstaff', 'flame', 'flashlight', 'floor', 'flower', 'fly', 'foam', 
            'food', 'footbridge', 'forceps', 'fork', 'forklift', 'fountain', 'fox', 
            'frame', 'fridge', 'frog', 'fruit', 'funnel', 'furnace', 'game controller', 
            'game machine', 'gas cylinder', 'gas hood', 'gas stove', 'gift box', 'glass', 'glass marble', 
            'globe', 'glove', 'goal', 'grandstand', 'grass', 'gravestone', 'ground', 
            'guardrail', 'guitar', 'gun', 'hammer', 'hand cart', 'handle', 'handrail', 
            'hanger', 'hard disk drive', 'hat', 'hay', 'headphone', 'heater', 'helicopter', 
            'helmet', 'holder', 'hook', 'horse', 'horse-drawn carriage', 'hot-air balloon', 'hydrovalve', 
            'ice', 'inflator pump', 'ipod', 'iron', 'ironing board', 'jar', 'kart', 
            'kettle', 'key', 'keyboard', 'kitchen range', 'kite', 'knife', 'knife block', 
            'ladder', 'ladder truck', 'ladle', 'laptop', 'leaves', 'lid', 'life buoy', 
            'light', 'light bulb', 'lighter', 'line', 'lion', 'lobster', 'lock', 
            'machine', 'mailbox', 'mannequin', 'map', 'mask', 'mat', 'match book', 
            'mattress', 'menu', 'metal', 'meter box', 'microphone', 'microwave', 'mirror', 
            'missile', 'model', 'money', 'monkey', 'mop', 'motorbike', 'mountain', 
            'mouse', 'mouse pad', 'musical instrument', 'napkin', 'net', 'newspaper', 'oar', 
            'ornament', 'outlet', 'oven', 'oxygen bottle', 'pack', 'pan', 'paper', 
            'paper box', 'paper cutter', 'parachute', 'parasol', 'parterre', 'patio', 'pelage', 
            'pen', 'pen container', 'pencil', 'person', 'photo', 'piano', 'picture', 
            'pig', 'pillar', 'pillow', 'pipe', 'pitcher', 'plant', 'plastic', 
            'plate', 'platform', 'player', 'playground', 'pliers', 'plume', 'poker', 
            'poker chip', 'pole', 'pool table', 'postcard', 'poster', 'pot', 'pottedplant', 
            'printer', 'projector', 'pumpkin', 'rabbit', 'racket', 'radiator', 'radio', 
            'rail', 'rake', 'ramp', 'range hood', 'receiver', 'recorder', 'recreational machines', 
            'remote control', 'road', 'robot', 'rock', 'rocket', 'rocking horse', 'rope', 
            'rug', 'ruler', 'runway', 'saddle', 'sand', 'saw', 'scale', 
            'scanner', 'scissors', 'scoop', 'screen', 'screwdriver', 'sculpture', 'scythe', 
            'sewer', 'sewing machine', 'shed', 'sheep', 'shell', 'shelves', 'shoe', 
            'shopping cart', 'shovel', 'sidecar', 'sidewalk', 'sign', 'signal light', 'sink', 
            'skateboard', 'ski', 'sky', 'sled', 'slippers', 'smoke', 'snail', 
            'snake', 'snow', 'snowmobiles', 'sofa', 'spanner', 'spatula', 'speaker', 
            'speed bump', 'spice container', 'spoon', 'sprayer', 'squirrel', 'stage', 'stair', 
            'stapler', 'stick', 'sticky note', 'stone', 'stool', 'stove', 'straw', 
            'stretcher', 'sun', 'sunglass', 'sunshade', 'surveillance camera', 'swan', 'sweeper', 
            'swim ring', 'swimming pool', 'swing', 'switch', 'table', 'tableware', 'tank', 
            'tap', 'tape', 'tarp', 'telephone', 'telephone booth', 'tent', 'tire', 
            'toaster', 'toilet', 'tong', 'tool', 'toothbrush', 'towel', 'toy', 
            'toy car', 'track', 'train', 'trampoline', 'trash bin', 'tray', 'tree', 
            'tricycle', 'tripod', 'trophy', 'truck', 'tube', 'turtle', 'tvmonitor', 
            'tweezers', 'typewriter', 'umbrella', 'unknown', 'vacuum cleaner', 'vending machine', 'video camera', 
            'video game console', 'video player', 'video tape', 'violin', 'wakeboard', 'wall', 'wallet', 
            'wardrobe', 'washing machine', 'watch', 'water', 'water dispenser', 'water pipe', 'water skate board', 
            'watermelon', 'whale', 'wharf', 'wheel', 'wheelchair', 'window', 'window blinds', 
            'wineglass', 'wire', 'wood', 'wool')

    PALETTE = None

    # def __init__(self, split, **kwargs):
    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.jpg',
            seg_map_suffix='.tif',
            ignore_index=-1,
            reduce_zero_label=False,
            int16=True,
            gt_seg_map_loader_cfg={"reduce_zero_label": False},
            **kwargs)
        # assert self.file_client.exists(self.img_dir) and self.split is not None