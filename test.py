from glob import glob
from tqdm import tqdm
import json

labels = glob("/mnt/haojun/itpsea4data/dataset/imagenet22k_azcopy/pa_pseudo_label_cam/*.png")
label_ids = [i.split("/")[-1].replace(".png", "") for i in labels]
print(len(label_ids))

with open("notebook/in21k_inter_ade_filter_v2.json") as f:
    in21k_inter_ade_filter_v2 = json.load(f)
    class_ids = list(in21k_inter_ade_filter_v2.keys())

label_ids = [i for i in label_ids if i[:i.find("_")] in class_ids]
# split = [l.split("/")[-1].replace(".png", "") for l in labels]
# with open("/mnt/haojun/resrchvc4data/dataset/imagenet22k_azcopy/pa_pseudo_label.txt", "w") as f:
#     for name in tqdm(split):
#         print(name, file=f)

# images = glob("/mnt/haojun/itpsea4data/dataset/imagenet22k_azcopy/jpg_images_ade_inter_in21k_by_wordnet/*.jpg")

# image_ids = [i.split("/")[-1].replace(".jpg", "") for i in images]


print(len(label_ids))
with open("/mnt/haojun/itpsea4data/dataset/imagenet22k_azcopy/in21k_inter_ade_v2_pseudo_label_cam.txt", "w") as f:
    for name in tqdm(label_ids):
        print(name, file=f)