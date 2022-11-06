export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,2,1,3
# export CUDA_LAUNCH_BLOCKING=1

# bash tools/dist_train.sh \
# configs/extend_voc_bce/bce_coseg_ade150w/vitb16_320k_wa150_c171_eval_a150.py

# bash tools/dist_test.sh \
# configs/large_voc/vit/visualization/large_voc_vitb16_cosine_coco171_eval_c171_slide.py \
# work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
# 4 \
# --eval mIoU

# python tools/test.py \
# configs/large_voc/vit/visualization/large_voc_vitb16_cosine_coco171_eval_c171_slide.py \
# work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
# --format-only \
# --eval-options "imgfile_prefix=./work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/pred/"

bash tools/dist_test.sh \
configs/extend_voc_bce/bce_coseg_ade150w/vitb16_320k_wa150_c171_eval_a150.py \
work_dirs/20221104_vitb16_320k_wa150_c171_ab4.0_co_ac2.0_wbce/iter_312000.pth \
4 \
--eval mIoU
# --format-only \
# --eval-options "imgfile_prefix=./work_dirs/20221104_vitb16_320k_wa150_c171_ab4.0_co_ac2.0_wbce/pred_c171/"
# --eval mIoU
# --show \
# --show-dir "./work_dirs/20221104_vitb16_320k_wa150_c171_ab4.0_co_ac2.0_wbce/vis_pred"

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


# amlt remove exp :20221105_vitb16_320k_wa847_c171_ab160.0_s100_wbce :20221105_vitb16_320k_wa150_c171_ab80.0_wbce :20221105_vitb16_320k_wa150_c171_ab40.0_wbce :20221105_vitb16_320k_wa150_c171_ab160.0_wbce :20221105_vitb16_320k_wa847_c171_ab80.0_s100_wbce :20221105_vitb16_320k_wa847_c171_ab40.0_s100_wbce