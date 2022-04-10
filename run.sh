
sudo bash local.sh \
configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_finetune_lr_0.0001_prior_0.05_loss_weight_0.1_mix_batch_coco-stuff164k_imagenet21k_local.py

# sudo bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_in21k_prior_0.005_loss_weight_0.1_mix_batch_coco-stuff164k_imagenet21k_local.py


# sudo bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_in21k_prior_0.3_loss_weight_0.1_mix_batch_coco-stuff164k_imagenet21k_local.py

# sudo bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_in21k_prior_0.4_loss_weight_0.1_mix_batch_coco-stuff164k_imagenet21k_local.py

# sudo bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_in21k_prior_1.0_loss_weight_0.1_mix_batch_coco-stuff164k_imagenet21k_local.py


# sudo bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_in21k_prior_1.0_loss_weight_0.1_mix_batch_coco-stuff164k_imagenet21k_local.py

# sudo bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_in21k_prior_0.4_loss_weight_0.1_mix_batch_coco-stuff164k_imagenet21k_local.py
# sudo bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_0.1_lambda_1.0_coco-stuff164k_local.py


# sudo bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_80k_bs16_prior_1.0_lambda_0.0_coco-stuff164k_local.py
# --auto-resume

# sudo bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_160k_bs16_prior_0.1_lambda_0.0_coco-stuff164k_local.py

# sudo bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_160k_ade20k.py

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


# work_dirs/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_in21k_prior_0.01_loss_weight_0.1_mix_batch_coco-stuff164k_imagenet21k_local/iter_40000.pth
# work_dirs/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_in21k_prior_0.05_loss_weight_0.1_mix_batch_coco-stuff164k_imagenet21k_local
# work_dirs/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_mix_batch_coco-stuff164k_imagenet21k_local
# work_dirs/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_in21k_prior_0.2_loss_weight_0.1_mix_batch_coco-stuff164k_imagenet21k_local
# work_dirs/segmenter-propagate_vit-b16_512x512_80k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_512x512_prior_0.05_loss_weight_0.1_mix_batch_coco-stuff164k_imagenet21k_local

