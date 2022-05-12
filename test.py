# from glob import glob
# from tqdm import tqdm
# import json

# labels = glob("/mnt/haojun/itpsea4data/dataset/imagenet22k_azcopy/pa_pseudo_label_cam/*.png")
# label_ids = [i.split("/")[-1].replace(".png", "") for i in labels]
# print(len(label_ids))

# with open("notebook/in21k_inter_ade_filter_v2.json") as f:
#     in21k_inter_ade_filter_v2 = json.load(f)
#     class_ids = list(in21k_inter_ade_filter_v2.keys())

# label_ids = [i for i in label_ids if i[:i.find("_")] in class_ids]
# # split = [l.split("/")[-1].replace(".png", "") for l in labels]
# # with open("/mnt/haojun/resrchvc4data/dataset/imagenet22k_azcopy/pa_pseudo_label.txt", "w") as f:
# #     for name in tqdm(split):
# #         print(name, file=f)

# # images = glob("/mnt/haojun/itpsea4data/dataset/imagenet22k_azcopy/jpg_images_ade_inter_in21k_by_wordnet/*.jpg")

# # image_ids = [i.split("/")[-1].replace(".jpg", "") for i in images]


# print(len(label_ids))
# with open("/mnt/haojun/itpsea4data/dataset/imagenet22k_azcopy/in21k_inter_ade_v2_pseudo_label_cam.txt", "w") as f:
#     for name in tqdm(label_ids):
#         print(name, file=f)

# import torch
# vitb16 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
# print(vitb16)

from glob import glob
import json
import numpy as np
import torch
import random
from tqdm import tqdm
import mmcv
from functools import partial
import os

cam_list = glob("/mnt/haojun2/dataset/imagenet22k_azcopy/annotations_cam_new/*.png")
print(len(cam_list))
with open("cam_list.txt", "a") as f:
    for c in cam_list:
        print(c, file=f)