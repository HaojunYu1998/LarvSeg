export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

bash tools/dist_train.sh \
configs/segmenter/segmenter-weak_vit-b16_160k_bs16_in21k_ade_all_no_attn_prior0.01.py

# bash tools/dist_test.sh \
# configs/segmenter/evaluation/baseline_no_attn_eval_ade_oracle.py \
# work_dirs/202209227_baseline_no_attn_160k_bs16_coco/iter_160000.pth \
# 4 \
# --eval mIoU


# sudo nvidia-docker run --ipc=host -it -v /mnt/haojun/itpsea4data:/itpsea4data --ipc=host hsfzxjy/mmseg:pytorch1.8.1-cuda10.2-cudnn7-devel /bin/bash
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