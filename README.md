# Training

1. clone the code to <local_path>

```git clone https://github.com/HaojunYuPKU/large_voc_seg```

2. pull the docker image

```sudo nvidia-docker run --ipc=host -it -v <local_path>:/workspace --ipc=host hsfzxjy/mmseg:pytorch1.8.1-cuda10.2-cudnn7-devel /bin/bash```

3. put all datasets in <local_path>/dataset/

```
<local_path>/
└── dataset/
      ├── ADEChallengeData2016/
            ├── images/
            ├── annotations/
      ├── ADE20K_2021_17_01/
      └── val.json
```

4. 

```
pip install -e .
```
