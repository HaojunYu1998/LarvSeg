export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_LAUNCH_BLOCKING=1



# bash tools/dist_train.sh \
# configs/coseg_ade150_pc459w/vitb16_320k_wp459_a150_pb0.1_sco_pc0.1_fg1_bg.py


# bash tools/dist_train.sh \
# configs/coseg_coco171_pc459w/vitb16_320k_wp459_c171_pb0.1_sco_pc0.1_fg1_bg.py

# bash tools/dist_train.sh \
# configs/baseline_coco171_pc459w/vitb16_320k_wp459_c171_pb0.1.py



# bash tools/dist_test.sh \
# configs/extend_voc/baseline_in11k/vitb16_640k_i11k_c171_s100_ib0.2_eval.py \
# work_dirs/20221107_vitb16_640k_i11k_c171_s100_ib0.2/iter_208000.pth \
# 4 \
# --eval mIoU

# bash tools/dist_test.sh \
# configs/extend_voc/baseline_in11k/vitb16_640k_i11k_c171_s100_ib0.2_eval_coinf.py \
# work_dirs/20221107_vitb16_640k_i11k_c171_s100_ib0.2/iter_208000.pth \
# 4 \
# --eval mIoU

# python tools/test.py \
# configs/extend_voc_bce/bce_coseg_ade150w_in124/vitb16_320k_wa150_c171_i124_ib8.0_ab8.0_co_ic4.0_ac4.0_wbce.py \
# work_dirs/20221106_vitb16_320k_wa150_c171_i124_ib8.0_ab8.0_co_ic4.0_ac4.0_wbce/iter_216000.pth \
# --format-only \
# --eval-options "imgfile_prefix=./work_dirs/20221106_vitb16_320k_wa150_c171_i124_ib8.0_ab8.0_co_ic4.0_ac4.0_wbce/pred/"

# python tools/test.py \
# configs/large_voc/vit/visualization/large_voc_vitb16_cosine_coco171_eval_c171_slide.py \
# work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
# --format-only \
# --eval-options "imgfile_prefix=./work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/pred/"

# bash tools/dist_test.sh \
# configs/extend_voc_bce/bce_coseg_ade150w/vitb16_320k_wa150_c171_eval_a150.py \
# work_dirs/20221104_vitb16_320k_wa150_c171_ab4.0_co_ac2.0_wbce/iter_312000.pth \
# 4 \
# --eval mIoU
# --format-only \
# --eval-options "imgfile_prefix=./work_dirs/20221104_vitb16_320k_wa150_c171_ab4.0_co_ac2.0_wbce/pred_c171/"
# --eval mIoU
# --show \
# --show-dir "./work_dirs/20221104_vitb16_320k_wa150_c171_ab4.0_co_ac2.0_wbce/vis_pred"

# sudo nvidia-docker run --ipc=host -it -v /mnt/haojun/itpsea4data:/workspace --ipc=host hsfzxjy/mmseg:pytorch1.8.1-cuda10.2-cudnn7-devel /bin/bash

# pip install -U amlt --extra-index-url https://msrpypi.azurewebsites.net/stable/leloojoo

# apt update
# apt-get install unzip -y
# apt-get install htop -y
# apt-get install tmux -y
# apt-get install vim -y
# pip install timm


