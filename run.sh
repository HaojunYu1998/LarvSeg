export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_LAUNCH_BLOCKING=1

bash tools/dist_train.sh \
configs/large_voc_v2/vit/attn4_cosine_in130/vitb16_attn4_cosine_160k_bs16_coco171_in130_seed0.1.py



# configs/large_voc_v2/vit/attn4_cosine_in130/vitb16_attn4_cosine_160k_bs16_coco171_in130_seed0.2.py
# configs/large_voc_v2/vit/attn4_cosine_in130/vitb16_attn4_cosine_160k_bs16_coco171_in130_seed0.3.py
# configs/large_voc_v2/vit/attn4_cosine_in130/vitb16_attn4_cosine_160k_bs16_coco171_in130_seed0.4.py


# configs/large_voc/vit/attn3_cosine_in130/large_voc_vitb16_cosine_160k_bs16_coco171_in130_eval_ade130_prior0.8.py


# bash tools/dist_test.sh \
# configs/large_voc/vit/visualization/large_voc_vitb16_attn6_cosine_vis_in_seed_prior0.9_max10000.py \
# work_dirs/20221013_large_voc_vitb16_cosine_160k_bs16_coco171_in130_eval_ade130_prior_loss_max10000/iter_160000.pth \
# 4 \
# --eval mIoU

# bash tools/dist_test.sh \
# configs/large_voc/vit/visualization/large_voc_vitb16_prop3_cosine_vis_in_seed_prior0.9_max10000.py \
# work_dirs/fix_bug_20221015_large_voc_vitb16_prop3_cosine_160k_bs16_coco171_in130_eval_ade130_thre0.2_detach_max10000/iter_160000.pth \
# 4 \
# --eval mIoU

# sudo nvidia-docker run --ipc=host -it -v /mnt/haojun/itpsea4data:/itpsea4data --ipc=host hsfzxjy/mmseg:pytorch1.8.1-cuda10.2-cudnn7-devel /bin/bash
# sudo nvidia-docker run --ipc=host -it -v /mnt/haojun/itpsea4data:/itpsea4data --ipc=host zeliu98/pytorch:superbench-nvcr21.05-fixfusedlamb-itp-mmcv-msrest /bin/bash

# pip install -U amlt --extra-index-url https://msrpypi.azurewebsites.net/stable/leloojoo
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