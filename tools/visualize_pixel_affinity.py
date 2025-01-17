import os

import numpy as np
import torch
import wandb
from tqdm import tqdm
import cv2
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt

# MaskFormer
from PIL import Image
from torch.nn import functional as F

a150_classes = (
    "wall",
    "building",
    "sky",
    "floor",
    "tree",
    "ceiling",
    "road",
    "bed",
    "windowpane",
    "grass",
    "cabinet",
    "sidewalk",
    "person",
    "earth",
    "door",
    "table",
    "mountain",
    "plant",
    "curtain",
    "chair",
    "car",
    "water",
    "painting",
    "sofa",
    "shelf",
    "house",
    "sea",
    "mirror",
    "rug",
    "field",
    "armchair",
    "seat",
    "fence",
    "desk",
    "rock",
    "wardrobe",
    "lamp",
    "bathtub",
    "railing",
    "cushion",
    "base",
    "box",
    "column",
    "signboard",
    "chest of drawers",
    "counter",
    "sand",
    "sink",
    "skyscraper",
    "fireplace",
    "refrigerator",
    "grandstand",
    "path",
    "stairs",
    "runway",
    "case",
    "pool table",
    "pillow",
    "screen door",
    "stairway",
    "river",
    "bridge",
    "bookcase",
    "blind",
    "coffee table",
    "toilet",
    "flower",
    "book",
    "hill",
    "bench",
    "countertop",
    "stove",
    "palm",
    "kitchen island",
    "computer",
    "swivel chair",
    "boat",
    "bar",
    "arcade machine",
    "hovel",
    "bus",
    "towel",
    "light",
    "truck",
    "tower",
    "chandelier",
    "awning",
    "streetlight",
    "booth",
    "television receiver",
    "airplane",
    "dirt track",
    "apparel",
    "pole",
    "land",
    "bannister",
    "escalator",
    "ottoman",
    "bottle",
    "buffet",
    "poster",
    "stage",
    "van",
    "ship",
    "fountain",
    "conveyer belt",
    "canopy",
    "washer",
    "plaything",
    "swimming pool",
    "stool",
    "barrel",
    "basket",
    "waterfall",
    "tent",
    "bag",
    "minibike",
    "cradle",
    "oven",
    "ball",
    "food",
    "step",
    "tank",
    "trade name",
    "microwave",
    "pot",
    "animal",
    "bicycle",
    "lake",
    "dishwasher",
    "screen",
    "blanket",
    "sculpture",
    "hood",
    "sconce",
    "vase",
    "traffic light",
    "tray",
    "ashcan",
    "fan",
    "pier",
    "crt screen",
    "plate",
    "monitor",
    "bulletin board",
    "shower",
    "radiator",
    "glass",
    "clock",
    "flag",
)

c171_classes = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "banner",
    "blanket",
    "branch",
    "bridge",
    "building-other",
    "bush",
    "cabinet",
    "cage",
    "cardboard",
    "carpet",
    "ceiling-other",
    "ceiling-tile",
    "cloth",
    "clothes",
    "clouds",
    "counter",
    "cupboard",
    "curtain",
    "desk-stuff",
    "dirt",
    "door-stuff",
    "fence",
    "floor-marble",
    "floor-other",
    "floor-stone",
    "floor-tile",
    "floor-wood",
    "flower",
    "fog",
    "food-other",
    "fruit",
    "furniture-other",
    "grass",
    "gravel",
    "ground-other",
    "hill",
    "house",
    "leaves",
    "light",
    "mat",
    "metal",
    "mirror-stuff",
    "moss",
    "mountain",
    "mud",
    "napkin",
    "net",
    "paper",
    "pavement",
    "pillow",
    "plant-other",
    "plastic",
    "platform",
    "playingfield",
    "railing",
    "railroad",
    "river",
    "road",
    "rock",
    "roof",
    "rug",
    "salad",
    "sand",
    "sea",
    "shelf",
    "sky-other",
    "skyscraper",
    "snow",
    "solid-other",
    "stairs",
    "stone",
    "straw",
    "structural-other",
    "table",
    "tent",
    "textile-other",
    "towel",
    "tree",
    "vegetable",
    "wall-brick",
    "wall-concrete",
    "wall-other",
    "wall-panel",
    "wall-stone",
    "wall-tile",
    "wall-wood",
    "water-other",
    "waterdrops",
    "window-blind",
    "window-other",
    "wood",
)

processed_c171_classes = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "television receiver",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "banner",
    "blanket",
    "branch",
    "bridge",
    "building",
    "bush",
    "cabinet",
    "cage",
    "cardboard",
    "carpet",
    "ceiling",
    "ceiling",
    "cloth",
    "clothes",
    "clouds",
    "counter",
    "cupboard",
    "curtain",
    "desk",
    "dirt track",
    "door",
    "fence",
    "floor",
    "floor",
    "floor",
    "floor",
    "floor",
    "flower",
    "fog",
    "food",
    "fruit",
    "furniture",
    "grass",
    "gravel",
    "ground",
    "hill",
    "house",
    "leaves",
    "light",
    "mat",
    "metal",
    "mirror",
    "moss",
    "mountain",
    "mud",
    "napkin",
    "net",
    "paper",
    "pavement",
    "pillow",
    "plant",
    "plastic",
    "platform",
    "playingfield",
    "railing",
    "railroad",
    "river",
    "road",
    "rock",
    "roof",
    "rug",
    "salad",
    "sand",
    "sea",
    "shelf",
    "sky",
    "skyscraper",
    "snow",
    "solid",
    "stairs",
    "stone",
    "straw",
    "structural",
    "table",
    "tent",
    "textile",
    "towel",
    "tree",
    "vegetable",
    "wall",
    "wall",
    "wall",
    "wall",
    "wall",
    "wall",
    "wall",
    "water",
    "waterdrops",
    "blind",
    "windowpane",
    "wood",
)


def main(
    embedding_dir,
    image_dir,
    gt_dir,
    out_dir,
    ignore_label=255,
    class_names=[],
    permute=False,
    device=0,
    reduce_zero=True,
    gt_suffix=".png",
    novel_classes=[],
    novel_only=False,
):
    embedding_files = os.listdir(embedding_dir)
    os.makedirs(out_dir, exist_ok=True)
    for emb_file in tqdm(embedding_files):
        gt_path = os.path.join(
            gt_dir, os.path.basename(emb_file).replace(".pth", gt_suffix)
        )
        label = np.asarray(Image.open(gt_path))
        if reduce_zero:
            label = (label - 1).astype(np.uint8)
        unique_label = np.unique(label)
        unique_label = unique_label[unique_label != ignore_label].tolist()
        class_list = [class_names[l] for l in unique_label]
        novel_classes_in_image = list(set(novel_classes) & set(class_list))
        novel_label = [class_names.index(c) for c in novel_classes_in_image]
        label_to_vis = novel_label if novel_only else unique_label
        class_to_vis = novel_classes_in_image if novel_only else class_list
        if novel_only and len(novel_classes_in_image) < 4:
            continue
        save_name = os.path.basename(emb_file).replace(".pth", "")
        save_name = save_name + "_" + "_".join(class_to_vis) + ".png"
        if os.path.exists(os.path.join(out_dir, save_name)):
            continue
        try:
            image_path = os.path.join(
                image_dir, os.path.basename(emb_file).replace(".pth", ".jpg")
            )
            image = np.asarray(Image.open(image_path))
            h, w, _ = image.shape
            embedding = (
                torch.load(os.path.join(embedding_dir, emb_file), map_location="cpu")
                .float()
                .cuda(device)
            )
            if permute:
                embedding = embedding.permute(0, 3, 1, 2)
            embedding = F.interpolate(
                embedding, size=(h, w), mode="bilinear", align_corners=True
            )
            embedding = embedding.reshape(-1, h * w).permute(1, 0)
            vis_img_list = []
            image_vis = Visualizer(image.copy())
            for l in label_to_vis:
                mask = label == l
                inds = np.nonzero(mask.reshape((-1,)))[0]
                inds = np.random.choice(inds, size=1)
                selected_embedding = embedding[torch.from_numpy(inds).cuda(device)]
                sim = F.cosine_similarity(
                    selected_embedding[:, None, :], embedding[None, ...], dim=-1
                ).reshape(-1, h, w)
                coords = np.stack([inds % w, inds // w], axis=1)  # k,2
                for one_sim, coord in zip(sim, coords):
                    one_sim = one_sim.cpu().numpy()
                    heatmap = (one_sim - np.min(one_sim)) / (
                        np.max(one_sim) - np.min(one_sim)
                    )
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv2.cvtColor(
                        cv2.applyColorMap(heatmap, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB
                    )
                    score_vis = Visualizer(image.copy())
                    # draw heatmap
                    score_vis.output.ax.imshow(heatmap, alpha=0.75)
                    score_vis.draw_circle([coord[0], coord[1]], radius=5, color="white")
                    image_vis.draw_circle([coord[0], coord[1]], radius=5, color="r")
                    score_img = score_vis.get_output().get_image()
                    vis_img_list.append(score_img)
            image = image_vis.get_output().get_image()
            num_imgs = len(vis_img_list) + 1
            f = plt.figure(figsize=(num_imgs * 5, 5))
            f.add_subplot(1, num_imgs, 1)
            plt.axis("off")
            plt.imshow(image)
            for i, vis_img in enumerate(vis_img_list):
                f.add_subplot(1, num_imgs, i + 2)
                plt.axis("off")
                plt.imshow(vis_img)
            # save_name = os.path.basename(emb_file).replace(".pth", "")
            # save_name = save_name + "_" + "_".join(class_list) + ".png"
            plt.savefig(os.path.join(out_dir, save_name))
            plt.close()
        except:
            continue


if __name__ == "__main__":
    a150_image_dir = "/itpsea4data/dataset/ADEChallengeData2016/images/validation/"
    a150_gt_dir = "/itpsea4data/dataset/ADEChallengeData2016/annotations/validation/"
    c171_image_dir = "/itpsea4data/dataset/coco_stuff164k/images/validation/"
    c171_gt_dir = "/itpsea4data/dataset/coco_stuff164k/annotations/validation/"
    # main(
    #     embedding_dir="/itpsea4data/OpenVocSeg/outputs/CLIP_RN50x64_embedding_ADE20K/",
    #     image_dir=a150_image_dir,
    #     gt_dir=a150_gt_dir,
    #     out_dir="vis_dirs/denseclip_a150/",
    #     ignore_label=255,
    #     class_names=a150_classes,
    #     permute=True,
    #     device=0,
    # )
    # main(
    #     embedding_dir="vis_dirs/baseline_c171_eval_a150_emb/",
    #     image_dir=a150_image_dir,
    #     gt_dir=a150_gt_dir,
    #     out_dir="vis_dirs/baseline_c171_eval_a150_novel_only/",
    #     ignore_label=255,
    #     class_names=a150_classes,
    #     novel_classes=list(set(a150_classes) - set(processed_c171_classes)),
    #     novel_only=True,
    #     device=1,
    # )
    main(
        embedding_dir="vis_dirs/baseline_c171_eval_c171_emb/",
        image_dir=c171_image_dir,
        gt_dir=c171_gt_dir,
        out_dir="vis_dirs/baseline_c171_eval_c171/",
        ignore_label=255,
        class_names=c171_classes,
        device=2,
        reduce_zero=False,
        gt_suffix="_labelTrainIds.png",
    )
