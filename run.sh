
sudo bash local.sh \
configs/segmenter/segmenter-propagate_vit-b16_512x512_160k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_prior_0.05_loss_weight_0.1_mix_batch_coco-stuff164k_imagenet21k_local.py


# sudo bash local_test.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_finetune_lr_0.0001_prior_0.05_loss_weight_0.1_mix_batch_coco-stuff164k_imagenet21k_local.py \
# work_dirs/segmenter-propagate_vit-b16_512x512_160k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_prior_0.05_loss_weight_0.1_mix_batch_coco-stuff164k_imagenet21k_local/iter_160000.pth \
# 4 \
# --eval mIoU


# sudo bash local_test.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_160k_bs16_prior_1.0_lambda_0.0_downsample_2_pseudo_labeling_in21k_ade_filter_prior_0.05_loss_weight_0.1_mix_batch_coco-stuff164k_imagenet21k_local.py \
# work_dirs/segmenter-propagate_vit-b16_512x512_160k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_prior_0.05_loss_weight_0.1_mix_batch_coco-stuff164k_imagenet21k_local/iter_160000.pth \
# 4 \
# --format-only \
# --eval-options "imgfile_prefix=work_dirs/segmenter-propagate_vit-b16_512x512_160k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_prior_0.05_loss_weight_0.1_mix_batch_coco-stuff164k_imagenet21k_local/inference_imagenet"
