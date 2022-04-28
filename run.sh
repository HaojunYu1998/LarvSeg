sudo bash local.sh \
configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_pseudo_label_vild_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local.py
# configs/segmenter/segmenter-propagate_vit-b16_512x512_320k_bs16_prior_1.0_lambda_0.0_downsample_1_in21k_ade_filter_prior_0.05_loss_weight_0.1_mix_batch_coco-stuff164k_imagenet21k_local.py



# # sudo bash local_test.sh \
# # configs/segmenter/segmenter-pixemb_vit-b16_512x512_160k_ade20k_eval_on_in21k.py \
# # work_dirs/segmenter-pixemb_vit-b16_512x512_160k_ade20k_2_samples_per_gpu/iter_160000.pth \
# # 4 \
# # --eval mIoU


# # sudo bash local_test.sh \
# # configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_in21k_ade_filter_vild_eval_on_in21k.py \
# # work_dirs/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local/iter_40000.pth \
# # 4 \
# # --format-only \
# # --eval-options "imgfile_prefix=work_dirs/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local/inference_imagenet"

# touch /mnt/data0/suspend.txt


# python -m torch.distributed.launch --nproc_per_node=8 --master_port=233333 \
# tools/train.py \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_320k_bs16_prior_1.0_lambda_0.0_downsample_1_in21k_ade_filter_prior_0.05_loss_weight_0.1_mix_batch_coco-stuff164k_imagenet21k.py \
# --launcher pytorch 

#"${ARGS[@]}"