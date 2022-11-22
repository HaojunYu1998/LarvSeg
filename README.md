# LarvSeg

This repository is an official implementation of the paper Exploring Image Classification Data For Large Vocabulary Semantic Segmentation via Category-wise Attentive Classifier

## Abstract

The vocabulary of current semantic segmentation models is limited because pixel-level labels are labor-intensive to obtain. On the contrary, image-level classification data owns much more images and semantic classes. In this paper, we propose to scale the vocabulary of semantic segmentation models with the help of classification data. The challenge lies in bridging the gap between image-level and pixel-level labels. Firstly, we propose a baseline to incorporate image-level supervision into the training process of a segmenter. Thus, it is able to perform segmentation on novel classes appeared in the classification data. To further bridge the gap, an intuitive idea is to extract the corresponding foreground regions of the image-level labels and attentively apply supervision to them. To this end, we investigate intra-class compactness of pixel features, which is important for precise region extraction. Surprisingly, we find that a model trained on segmentation data is able to group pixels of classes outside the training vocabulary. Inspired by this observation, we propose a category-wise attentive classifier to adaptively highlight the foreground regions and suppress the background regions by mining cross-image semantics. The overall framework is called LarvSeg. Experimental results show that the simple baseline surpasses previous open vocabulary arts by a large margin. Moreover, LarvSeg significantly improves the baseline performance, especially on the classes with only image-level labels. Finally, for the first time, we provide a model to perform semantic segmentation on twenty-one thousand categories and qualitative results are provided. The code will be released soon.

## Citing LarvSeg

If you find LarvSeg useful in your research, please consider citing:
```
```

## Usage

1. Pull the nvidia-docker.

```
sudo nvidia-docker run --ipc=host -it -v <local_path>:/workspace --ipc=host hsfzxjy/mmseg:pytorch1.8.1-cuda10.2-cudnn7-devel /bin/bash
```

For exsample,

```
sudo nvidia-docker run --ipc=host -it -v /mnt/haojun/itpsea4data:/workspace --ipc=host hsfzxjy/mmseg:pytorch1.8.1-cuda10.2-cudnn7-devel /bin/bash
```

2. Clone the git repo to /workspace/large_voc_seg

```
cd /workspace
git clone https://github.com/HaojunYuPKU/large_voc_seg
```

3. Download the pretrained backbone.

```
cd /workspace/large_voc_seg/pretrain
wget https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz
```

4. Manage all datasets in /workspace/dataset/ as the following format (by soft link)

```
/workspace/
└── dataset/
      ├── I22K/
            ├── fall11_whole/
                  ├── <category_id>/*.JPEG
      ├── A150/
            ├── images/
                  ├── training/*.jpg
                  ├── validation/*.jpg
            ├── annotations/
                  ├── training/*.png
                  ├── validation/*.png
      ├── A847/
            ├── images/
                  ├── training/*.jpg
                  ├── validation/*.jpg
            ├── annotations/
                  ├── training/*.tif
                  ├── validation/*.tif
      └── C171/
            ├── images/
                  ├── training/*.jpg
                  ├── validation/*.jpg
            ├── annotations/
                  ├── training/*.png
                  ├── validation/*.png
```

5. Install the LarvSeg package in developing mode

```
pip install -e .
```

6. Training command:

```
bash tools/dist_train.sh <path_to_config>
```

7. Evaluation command:

```
bash tools/dist_test.sh <path_to_config> <path_to_checkpoint> <num_gpus> --eval mIoU
```
