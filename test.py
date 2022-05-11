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

import torch


cls_emb1 = torch.load("pretrain/cls_emb_in21k_all_vild_split1.pth")
cls_emb2 = torch.load("pretrain/cls_emb_in21k_all_vild_split2.pth")
cls_emb3 = torch.load("pretrain/cls_emb_in21k_all_vild_split3.pth")
cls_emb4 = torch.load("pretrain/cls_emb_in21k_all_vild_split4.pth")
print(cls_emb1.shape,cls_emb2.shape,cls_emb3.shape,cls_emb4.shape)

cls_emb = torch.cat([cls_emb1,cls_emb2,cls_emb3,cls_emb4 ], dim=0)
del cls_emb1, cls_emb2, cls_emb3, cls_emb4

import json
with open("notebook/in21k_class_names_with_definition_dict_more_than_500.json") as f:
    in21k_class_names_with_definition = json.load(f)

names = list(in21k_class_names_with_definition.keys())

a = {}
for t, n in zip(cls_emb, names):
    a[n] = t
torch.save(a, "cls_emb_in21k_all_dict.pth")

