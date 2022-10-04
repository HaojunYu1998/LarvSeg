export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_LAUNCH_BLOCKING=1

# bash tools/dist_train.sh \
# configs/large_voc/large_voc_swinv2b16_cosine_80k_bs16_cocostuff_ade847_temp0.05.py

bash tools/dist_train.sh \
configs/large_voc/swinv2/large_voc_swinv2b16_cosine_80k_bs16_cocostuff_ade847_structure_weight1.0_thre0.0.py


# configs/large_voc/large_voc_swinv2b16_cosine_80k_bs16_cocostuff_ade847_temp0.05_max200.py
# configs/large_voc/large_voc_swinv2b16_cosine_80k_bs16_cocostuff_ade847_temp0.05_max500.py
# configs/large_voc/large_voc_swinv2b16_cosine_80k_bs16_cocostuff_ade847_temp0.05_max1000.py


# bash tools/dist_test.sh \
# configs/segmenter/segmenter-cosine_vit-b16_80k_bs16_base_config_ade847.py \
# work_dirs/segmenter-cosine_vit-b16_80k_bs16_base_config_ade847/iter_80000.pth \
# 4 \
# --eval mIoU


# sudo nvidia-docker run --ipc=host -it -v /mnt/haojun/itpsea4data:/itpsea4data --ipc=host hsfzxjy/mmseg:pytorch1.8.1-cuda10.2-cudnn7-devel /bin/bash
# sudo nvidia-docker run --ipc=host -it -v /mnt/haojun/itpsea4data:/itpsea4data --ipc=host zeliu98/pytorch:superbench-nvcr21.05-fixfusedlamb-itp-mmcv-msrest /bin/bash

# pip install -U amlt --extra-index-url https://msrpypi.azurewebsites.net/stable/7e404de797f4e1eeca406c1739b00867 --extra-index-url https://azuremlsdktestpypi.azureedge.net/K8s-Compute/D58E86006C65
# apt update
# apt-get install unzip -y
# apt-get install htop -y
# apt-get install tmux -y
# apt-get install vim -y
# pip install install git+https://github.com/lucasb-eyer/pydensecrf.git
# cd third_party/CLIP
# pip install -e .
# cd ../detectron2
# pip install -e .
# cd ../..
# mkdir -p /mnt/haojun2
# ln -s /itpsea4data/dataset /mnt/haojun2/dataset
# pip install timm