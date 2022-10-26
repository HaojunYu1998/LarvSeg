import os
from glob import glob
from PIL import Image
import numpy as np
import torch
root_dir = "/itpsea4data/mmseg/work_dirs/20221019_vitb16_cosine_160k_bs16_coco171_in130_avgpool"
score_files = glob(os.path.join(root_dir, "cls_score_map20", "*.pth"))
print(len(score_files))

def form_dict(files):
    file_dict = {}
    for f in files:
        name = f.split("/")[-1].split("_")[0]
        file = f.split("/")[-1].split(".")[0]
        if name not in file_dict:
            file_dict[name] = []
        file_dict[name].append(file)
    return file_dict

def get_image(file):
    img_path = os.path.join(root_dir, f"images20/{file}.png")
    return np.array(Image.open(img_path))

def get_embed(file):
    emb_path = os.path.join(root_dir, f"embedding20/{file}.pth")
    return torch.load(emb_path, map_location="cpu")

def get_score(file):
    score_path = os.path.join(root_dir, f"cls_score_map20/{file}.pth")
    return torch.load(score_path, map_location="cpu")

def coseg_map(file, sup_files, num_seed=40):
    sup_pixels = []
    for sup_file in sup_files:
        score = get_score(sup_file)
        embed = get_embed(sup_file)
        # print(score.shape, embed.shape)
        inds = score.flatten().topk(num_seed).indices
        sup_pixels.append(embed.reshape(768, -1).permute(1,0)[inds])
    sup_pixels = torch.cat(sup_pixels)
    embed = get_embed(file)
    H, W = embed.shape[-2:]
    embed = embed.reshape(768, -1).permute(1,0)
    embed = embed / embed.norm(dim=-1, keepdim=True)
    sup_pixels = sup_pixels / sup_pixels.norm(dim=-1, keepdim=True)
    cos_mat = embed @ sup_pixels.T
    score = cos_mat.mean(dim=-1)
    return score.reshape(H, W)

file_dict = form_dict(score_files)
for name in file_dict:
    print(name)
    for file in file_dict[name]:
        img = get_image(file)
        score = coseg_map(file, file_dict[name])
        import matplotlib.pyplot as plt
        f = plt.figure()
        f.add_subplot(1, 2, 1)
        plt.axis("off")
        plt.imshow(img)
        f.add_subplot(1, 2, 2)
        plt.axis("off")
        plt.imshow(score)
        plt.savefig(os.path.join(root_dir, f"coseg/{file}.png"))
        plt.close()