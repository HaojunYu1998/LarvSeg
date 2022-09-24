export OMP_NUM_THREADS=1


sudo bash local.sh \
configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_downsample_2_base_config_in21k_ade_full_merged_vild.py

sudo bash local_test.sh \
configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_downsample_2_base_config_in21k_ade_full_merged_vild.py \
work_dirs/segmenter-propagate_vit-b16_512x512_40k_bs16_downsample_2_base_config_in21k_ade_full_merged_vild/iter_4000.pth \
4 \
--eval mIoU