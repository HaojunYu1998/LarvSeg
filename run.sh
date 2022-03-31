
sudo bash local.sh \
configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_mix_batch_coco-stuff164k_imagenet21k_local.py


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
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_0.1_lambda_1.0_mix_batch_coco-stuff164k_ade20k_local.py \
# work_dirs/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_0.1_lambda_1.0_coco-stuff164k_local/iter_40000.pth \
# 1 \
# --format-only \
# --eval-options "imgfile_prefix=work_dirs/inference"



# sudo bash local_test.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_0.1_lambda_1.0_mix_batch_coco-stuff164k_ade20k_local.py \
# work_dirs/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_0.1_lambda_1.0_mix_batch_coco-stuff164k_ade20k_local/iter_40000.pth \
# 4 \
# --eval mIoU

# 4 \
# --format-only \
# --eval-options "imgfile_prefix=work_dirs/segmenter-pixemb_vit-b16_512x512_160k_ade20k_2_samples_per_gpu/inference"