# sudo bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_downsample_2_pairwise_affinity_cam_thre_0.7_pa_thre_0.9_in21k_ade_filter_vild_v2_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local.py \
# --auto-resume
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_downsample_2_pairwise_affinity_cam_thre_0.9_pa_thre_0.95_in21k_ade_filter_vild_v2_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local.py
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_v2_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local.py
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_v2_prior_0.05_loss_weight_0.1_mix_batch_coco-stuff164k_imagenet21k_local.py

# sudo bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs8_downsample_4_base_config_prompt_learning_shape_16_0_lrm_1e-1_wd_1e-1_local.py


# sudo bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_downsample_2_base_config_prior_loss_weight_0.2_self_train_thresh_210.py


# sudo bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_downsample_2_base_config_prior_loss_weight_0.1_self_train_thresh_210.py


# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_downsample_2_base_config_pseudo_labeling_local.py

# sudo bash local_test.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_10k_bs16_downsample_2_base_config_lr_1e-3_pseudo_label_cam_tune_from_sota_loss_weight_0.05_thresh_0.9_rr1.py \
# work_dirs/segmenter-propagate_vit-b16_512x512_320k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_v2_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local/iter_224000.pth \
# 4 \
# --eval mIoU

# echo 'zhzx20140317' | sudo -S bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_10k_bs16_downsample_2_base_config_lr_1e-4_pseudo_label_cam_tune_from_sota_loss_weight_0.1_thresh_0.9_rr1.py

# sleep 10

# echo 'zhzx20140317' | sudo -S bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_10k_bs16_downsample_2_base_config_lr_1e-4_pseudo_label_cam_tune_from_sota_loss_weight_0.5_thresh_0.9_rr1.py

# sleep 10

# echo 'zhzx20140317' | sudo -S bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_10k_bs16_downsample_2_base_config_lr_1e-3_pseudo_label_cam_tune_from_sota_loss_weight_0.05_thresh_0.9_rr1.py

# sleep 10

# echo 'zhzx20140317' | sudo -S bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_10k_bs16_downsample_2_base_config_lr_1e-4_pseudo_label_cam_tune_from_sota_loss_weight_0.05_thresh_0.9_rr1.py

# sleep 10

# echo 'zhzx20140317' | sudo -S bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_10k_bs16_downsample_2_base_config_lr_1e-5_pseudo_label_cam_tune_from_sota_loss_weight_0.05_thresh_0.9_rr1.py

# sudo bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_320k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_v2_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local.py \
# --auto-resume

sudo bash local_test.sh \
configs/segmenter/segmenter-propagate_vit-b16_512x512_eval_on_ade_full.py \
work_dirs/segmenter-propagate_vit-b16_512x512_320k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_v2_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local/iter_224000.pth \
3 \
--eval mIoU


# sudo bash local_test.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_eval_on_ade_full.py \
# work_dirs/segmenter-propagate_vit-b16_512x512_320k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_v2_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local/iter_320000.pth \
# 4 \
# --format-only \
# --eval-options "imgfile_prefix=work_dirs/segmenter-propagate_vit-b16_512x512_320k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_v2_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local/inference_imagenet"
