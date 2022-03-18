# sudo ba/512_80k_coco-stuff164k.py

sudo bash local.sh \
configs/segmenter/segmenter-propagate_vit-b16_512x512_160k_coco-stuff164k.py

# sudo bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_160k_ade20k.py

# sudo bash local_test.sh \
# configs/segmenter/segmenter-pixemb_vit-b16_512x512_160k_ade20k_test_whole_thresh_pixemb_unary.py \
# work_dirs/segmenter-pixemb_vit-b16_512x512_160k_ade20k_2_samples_per_gpu/iter_160000.pth \
# 4 \
# --eval mIoU

# 4 \
# --format-only \
# --eval-options "imgfile_prefix=work_dirs/segmenter-pixemb_vit-b16_512x512_160k_ade20k_2_samples_per_gpu/inference"