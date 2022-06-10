export OMP_NUM_THREADS=1

sudo bash local.sh \
configs/segmenter/assignment_segmenter_vit-b16_80k_voc2012.py


sudo bash local_test.sh \
configs/segmenter/assignment_segmenter_swin_nano_80k_voc2012.py \
work_dirs/assignment_segmenter_swin_nano_80k_voc2012/latest.pth \
4 \
--format-only \
--eval-options "imgfile_prefix=work_dirs/assignment_segmenter_swin_nano_80k_voc2012/predictions"