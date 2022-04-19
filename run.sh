# echo 'zhzx20140317' | sudo -S bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_base_config_structure_loss_weight_10.0_temp_0.07_local.py

echo 'zhzx20140317' | sudo -S bash local.sh \
configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_base_config_structure_loss_weight_10.0_temp_0.4_local.py

# echo 'zhzx20140317' | sudo -S bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_base_config_structure_loss_weight_40.0_temp_0.4_local.py

# echo 'zhzx20140317' | sudo -S bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_base_config_structure_loss_weight_10.0_temp_1.0_local.py


# sudo bash local_test.sh \
# configs/segmenter/segmenter-pixemb_vit-b16_512x512_160k_ade20k_eval_on_in21k.py \
# work_dirs/segmenter-pixemb_vit-b16_512x512_160k_ade20k_2_samples_per_gpu/iter_160000.pth \
# 4 \
# --eval mIoU


# sudo bash local_test.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_in21k_ade_filter_vild_eval_on_in21k.py \
# work_dirs/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local/iter_40000.pth \
# 4 \
# --format-only \
# --eval-options "imgfile_prefix=work_dirs/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local/inference_imagenet"
