import os

import numpy as np
import torch
import wandb
from tqdm import tqdm
import cv2
from detectron2.utils.visualizer import Visualizer

# MaskFormer
from PIL import Image
from torch.nn import functional as F
from torchvision.utils import make_grid


def main(
    embedding_dir,
    image_dir=None,
    gt_dir=None,
    ignore_label=255,
):
    # load embedding
    embedding_files = os.listdir(embedding_dir)
    # visualize
    for emb_file in tqdm(embedding_files):
        image_path = os.path.join(
            image_dir, os.path.basename(emb_file).replace(".pth", ".jpg")
        )
        gt_path = os.path.join(
            gt_dir, os.path.basename(emb_file).replace(".pth", ".png")
        )
        image = np.asarray(Image.open(image_path))
        h, w, _ = image.shape
        gt = np.asarray(Image.open(gt_path))
        embedding = torch.load(os.path.join(embedding_dir, emb_file)).float().cuda()
        embedding = F.interpolate(embedding, size=(h, w), mode="bilinear", align_corners=True)
        if embedding.dim() == 4:
            embedding = embedding.reshape(embedding.shape[1], -1)
        embedding = embedding.reshape(embedding.shape[0], -1).permute(1, 0)
        mask = image[:, :, 0] >= 0
        inds = np.nonzero(mask.reshape((-1,)))[0]
        inds = np.random.choice(inds, size=2)
        selected_embedding = embedding[torch.from_numpy(inds).cuda()]
        sim = F.cosine_similarity(
            selected_embedding[:, None, :], embedding[None, ...], dim=-1
        ).reshape(-1, h, w)
        coords = np.stack([inds % w, inds // w], axis=1)  # k,2
        for one_sim, coord in zip(sim, coords):
            one_sim = one_sim.cpu().numpy()
            heatmap = (one_sim - np.min(one_sim)) / (np.max(one_sim)-np.min(one_sim))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.cvtColor(
                cv2.applyColorMap(heatmap, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB
            )
            vis = Visualizer(image.copy())
            # draw heatmap
            vis.output.ax.imshow(heatmap, alpha=0.9)
            vis.draw_circle([coord[0], coord[1]], radius=5, color="r")
            vis.get_output().get_image()

if __name__ == "__main__":
    main(
        embedding_dir="/itpsea4data/OpenVocSeg/outputs/CLIP_RN50x64_embedding_ADE20K/",
        image_dir="/itesea4data/dataset/ADEChallengeData2016/images/validation/",
        gt_dir="/itesea4data/dataset/ADEChallengeData2016/annotations/validation/",
        ignore_label=255
    )
