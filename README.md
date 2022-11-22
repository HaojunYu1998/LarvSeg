# Training

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
