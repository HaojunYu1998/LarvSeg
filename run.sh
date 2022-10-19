export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_LAUNCH_BLOCKING=1

bash tools/dist_train.sh \
configs/large_voc_v2/vit/cosine_in130/vitb16_cosine_160k_bs16_coco171_in130_seed0.1_coseg0.2_memory1000_mean_valid.py

# configs/large_voc_v2/vit/cosine_in585/vitb16_cosine_160k_bs16_coco171_in585_seed0.1_coseg0.2_max_valid.py
# bash tools/dist_test.sh \
# configs/large_voc_v2/vit/visualization/vitb16_attn4_cosine_coseg_vis.py \
# work_dirs/20221018_vitb16_attn4_cosine_160k_bs16_coco171_in130_seed0.1_coseg0.4/iter_160000.pth \
# 4 \
# --eval mIoU

# sudo nvidia-docker run --ipc=host -it -v /mnt/haojun/itpsea4data:/itpsea4data --ipc=host hsfzxjy/mmseg:pytorch1.8.1-cuda10.2-cudnn7-devel /bin/bash
# sudo nvidia-docker run --ipc=host -it -v /mnt/haojun/itpsea4data:/itpsea4data --ipc=host zeliu98/pytorch:superbench-nvcr21.05-fixfusedlamb-itp-mmcv-msrest /bin/bash

# pip install -U amlt --extra-index-url https://msrpypi.azurewebsites.net/stable/leloojoo
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