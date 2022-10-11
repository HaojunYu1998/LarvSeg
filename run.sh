export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_LAUNCH_BLOCKING=1

# bash tools/dist_train.sh \
# configs/large_voc/vit/large_voc_vitb16_prop_head_cosine_80k_bs16_coco171_in130_eval_ade130.py

bash tools/dist_train.sh \
configs/large_voc/vit/large_voc_vitb16_prop_head_cosine_80k_bs16_coco171_in130_eval_ade130.py

# bash tools/dist_train.sh \
# configs/large_voc/vit/large_voc_vitb16_linear_80k_bs16_coco171_in585_eval_ade585_prior_loss.py

# bash tools/dist_test.sh \
# configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle1.py \
# work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_ade847_prior_loss/iter_80000.pth \
# 4 \
# --eval mIoU > 1coco171_prior_structure1_v1.txt

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