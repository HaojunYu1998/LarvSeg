touch /mnt/data0/suspend.txt

# pip install mmcv-full==1.3.12 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.9.0a0+2ecb2c7/index.html
# pip install -e .

export OMP_NUM_THREADS=1


python -m torch.distributed.launch \
--nproc_per_node=8 --master_port=233333 \
tools/test.py \
configs/segmenter/segmenter-propagate_vit-b16_512x512_eval_on_ade_full.py \
work_dirs/segmenter-propagate_vit-b16_512x512_320k_bs16_downsample_2_base_config_in21k_ade_full_merged_vild_all_500_rr1/iter_320000.pth \
--launcher pytorch \
--format-only \
--eval-options "imgfile_prefix=work_dirs/ablation_base_config_coco_in21k_adefull_hyper/perd_full_mask"

# python -m torch.distributed.launch \
# --nproc_per_node=8 --master_port=233333 \
# tools/test.py \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_eval_on_ade.py \
# work_dirs/segmenter-propagate_vit-b16_512x512_40k_bs_16_downsample_2_base_config_structure_contrastive_loss_weight_1.0_in21k_inter_ade_rr1/latest.pth \
# --eval mIoU \
# --launcher pytorch

# python -m torch.distributed.launch --nproc_per_node=8 --master_port=2333 \
# tools/train.py \
# configs/segmenter/ablation_base_config_ade_full_supervised.py \
# --launcher pytorch


# python -m torch.distributed.launch --nproc_per_node=8 --master_port=2333 \
# tools/train.py \
# configs/segmenter/ablation_base_config_bs2,8x8_coco_in21k_adefull_hyper.py \
# --launcher pytorch





# python -m torch.distributed.launch \
# --nproc_per_node=8 --master_port=233333 \
# tools/test.py \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_eval_on_ade_full_miou_multiscale.py \
# work_dirs/segmenter-propagate_vit-b16_512x512_320k_bs16_downsample_2_base_config_in21k_ade_full_merged_vild_all_500_rr1/iter_320000.pth \
# --eval mIoU \
# --launcher pytorch \
# --aug-test

# python -m torch.distributed.launch \
# 	--nproc_per_node=8 --master_port=233333 \
# 	tools/test.py \
# 	configs/segmenter/segmenter-propagate_vit-b16_512x512_eval_on_ade_full_multiscale.py \
# 	work_dirs/ablation_base_config_ade_full_supervised/iter_140000.pth \
# 	--eval mIoU \
# 	--launcher pytorch \
# 	--aug-test

# python -m torch.distributed.launch \
# 	--nproc_per_node=8 --master_port=233333 \
# 	tools/test.py \
# 	configs/segmenter/segmenter-propagate_vit-b16_512x512_eval_on_ade_full_hyper_multiscale.py \
# 	work_dirs/ablation_base_config_ade_full_supervised/iter_140000.pth \
# 	--eval mIoU \
# 	--launcher pytorch \
# 	--aug-test
