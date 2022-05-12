# sudo bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_v2_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local.py
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_v2_prior_0.05_loss_weight_0.1_mix_batch_coco-stuff164k_imagenet21k_local.py
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_pseudo_label_cam_vild_v2_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local.py

# # sudo bash local_test.sh \
# # configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_in21k_ade_filter_vild_eval_on_in21k.py \
# # work_dirs/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local/iter_40000.pth \
# # 4 \
# # --format-only \
# # --eval-options "imgfile_prefix=work_dirs/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local/inference_imagenet"

touch /mnt/data0/suspend.txt

# pip install mmcv-full==1.3.12 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.9.0a0+2ecb2c7/index.html
# pip install -e .

export OMP_NUM_THREADS=1


python -m torch.distributed.launch --nproc_per_node=8 --master_port=233333 \
tools/train.py \
configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_downsample_2_base_config_in21k_bases_500_fold0_rr1.py \
--launcher pytorch

# python -m torch.distributed.launch --nproc_per_node=8 --master_port=233335 \
# tools/train.py \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_10k_bs16_downsample_2_base_config_prompt_learning_sample32_shape_16_0_lr_1e-3_wd_0_rr1.py \
# --launcher pytorch

# sleep 10

# python -m torch.distributed.launch --nproc_per_node=8 --master_port=233335 \
# tools/train.py \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_10k_bs16_downsample_2_base_config_prompt_learning_sample32_shape_16_0_lr_1e-5_wd_0_rr1.py \
# --launcher pytorch

# sleep 10


# python -m torch.distributed.launch \
# --nproc_per_node=8 --master_port=233333 \
# tools/test.py \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_eval_on_in21k.py \
# pretrain/iter_224000.pth \
# --launcher pytorch \
# --format-only \
# --eval-options "imgfile_prefix=work_dirs/segmenter-propagate_vit-b16_512x512_eval_on_in21k/inference_imagenet"
