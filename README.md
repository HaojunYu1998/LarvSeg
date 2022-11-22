# LarvSeg

This repository is an official implementation of the paper Exploring Image Classification Data For Large Vocabulary Semantic Segmentation via Category-wise Attentive Classifier

## Abstract

The vocabulary of current semantic segmentation models is limited because pixel-level labels are labor-intensive to obtain. On the contrary, image-level classification data owns much more images and semantic classes. In this paper, we propose to scale the vocabulary of semantic segmentation models with the help of classification data. The challenge lies in bridging the gap between image-level and pixel-level labels. Firstly, we propose a baseline to incorporate image-level supervision into the training process of a segmenter. Thus, it is able to perform segmentation on novel classes appeared in the classification data. To further bridge the gap, an intuitive idea is to extract the corresponding foreground regions of the image-level labels and attentively apply supervision to them. To this end, we investigate intra-class compactness of pixel features, which is important for precise region extraction. Surprisingly, we find that a model trained on segmentation data is able to group pixels of classes outside the training vocabulary. Inspired by this observation, we propose a category-wise attentive classifier to adaptively highlight the foreground regions and suppress the background regions by mining cross-image semantics. The overall framework is called LarvSeg. Experimental results show that the simple baseline surpasses previous open vocabulary arts by a large margin. Moreover, LarvSeg significantly improves the baseline performance, especially on the classes with only image-level labels. Finally, for the first time, we provide a model to perform semantic segmentation on twenty-one thousand categories and qualitative results are provided. The code will be released soon.

## Citing LarvSeg

If you find LarvSeg useful in your research, please consider citing:
```
```

## Usage

1. clone the code to <local_path>

```git clone https://github.com/HaojunYuPKU/large_voc_seg```

2. pull the docker image

```sudo nvidia-docker run --ipc=host -it -v <local_path>:/workspace --ipc=host hsfzxjy/mmseg:pytorch1.8.1-cuda10.2-cudnn7-devel /bin/bash```

3. put all datasets in /workspace/dataset/

```
/workspace/
└── dataset/
      ├── I22K/
            ├── fall11_whole/
                  ├── <category_id>/*.JPEG
      ├── A150/
            ├── images/
                  ├── train/*.jpg
                  ├── validation/*.jpg
            ├── annotations/
                  ├── train/*.png
                  ├── validation/*.png
      ├── A847/
            ├── images/
                  ├── train/*.jpg
                  ├── validation/*.jpg
            ├── annotations/
                  ├── train/*.tif
                  ├── validation/*.tif
      └── C171/
            ├── images/
                  ├── train/*.jpg
                  ├── validation/*.jpg
            ├── annotations/
                  ├── train/*.png
                  ├── validation/*.png
```

4. install mmseg package

```
pip install -e .
```

5. training command

```
bash tools/dist_train.sh \
<path_to_config>
```

6. evaluation command

```
bash tools/dist_test.sh \
<path_to_config> \
<path_to_checkpoint> \
<num_gpus> \
--eval mIoU
```
