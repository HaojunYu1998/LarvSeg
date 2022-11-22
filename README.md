# Training

1. clone the code to <local_path>

```git clone https://github.com/HaojunYuPKU/large_voc_seg```

2. pull the docker image

```sudo nvidia-docker run --ipc=host -it -v <local_path>:/workspace --ipc=host hsfzxjy/mmseg:pytorch1.8.1-cuda10.2-cudnn7-devel /bin/bash```

3. put all datasets in /workspace/dataset/

```
/workspace/
└── dataset/
      ├── imagenet22k_azcopy/
            ├── fall11_whole/
                  ├── <category_id>/*.JPEG
      ├── ADEChallengeData2016/
            ├── images/
                  ├── train/*.jpg
                  ├── validation/*.jpg
            ├── annotations/
                  ├── train/*.png
                  ├── validation/*.png
      ├── ADE20K_2021_17_01/
            ├── images/
                  ├── train/*.jpg
                  ├── validation/*.jpg
            ├── annotations/
                  ├── train/*.tif
                  ├── validation/*.tif
      └── coco_stuff164k/
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
