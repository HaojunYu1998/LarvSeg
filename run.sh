export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_LAUNCH_BLOCKING=1

bash tools/dist_train.sh \
configs/extend_voc/peudo_baseline_ade150w/vitb16_320k_wa150_c171_ab0.1_ap0.1_pseudo.py


# bash tools/dist_test.sh \
# configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle1.py \
# work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_ade150_prior_structure_loss/iter_80000.pth \
# 4 \
# --eval mIoU

# python tools/test.py \
# configs/large_voc_v2/vit/cosine_in130/vitb16_cosine_160k_bs16_coco171_in130_avgpool.py \
# work_dirs/20221019_vitb16_cosine_160k_bs16_coco171_in130_avgpool/iter_160000.pth \
# --show \
# --show-dir "./work_dirs/20221019_vitb16_cosine_160k_bs16_coco171_in130_avgpool/vis_pred_tag"

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

# amlt remove exp :20221101_vitb16_320k_i124_c171_ib0.2_co_ic0.1_mbs20_wu20_fg40_bg5_bgt0.30_mse2.0 :20221101_vitb16_320k_wa150_c171_i124_ib0.2_ab0.2_wbce