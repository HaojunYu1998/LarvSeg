export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_LAUNCH_BLOCKING=1

bash tools/dist_train.sh \
configs/large_voc/vit/large_voc_vitb16_cosine_80k_bs16_cocostuff_ade847_temp0.05_max100_min10.py


# bash tools/dist_test.sh \
# configs/large_voc/vit/large_voc_vitb16_cosine_80k_bs16_cocostuff_ade847_temp0.05.py \
# work_dirs/20221004_large_voc_vitb16_cosine_80k_bs16_cocostuff_ade847_temp0.05_sing/iter_80000.pth \
# 4 \
# --eval mIoU > weak.txt

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