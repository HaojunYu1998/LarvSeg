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


# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_downsample_2_base_config_pseudo_labeling_local.py

# pip install mmcv-full==1.3.12 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.1/index.html

export OMP_NUM_THREADS=1


sudo bash local_test.sh \
configs/segmenter/assignment_segmenter_swin_nano_80k_voc2012.py \
work_dirs/assignment_segmenter_swin_nano_80k_voc2012/latest.pth \
4 \
--format-only \
--eval-options "imgfile_prefix=work_dirs/assignment_segmenter_swin_nano_80k_voc2012/predictions"


sudo bash local_test.sh \
configs/segmenter/assignment_fcn_swin_nano_80k_voc2012.py \
work_dirs/assignment_fcn_swin_nano_80k_voc2012/latest.pth \
4 \
--format-only \
--eval-options "imgfile_prefix=work_dirs/assignment_fcn_swin_nano_80k_voc2012/predictions"


sudo bash local_test.sh \
configs/segmenter/assignment_segmenter_vit-b16_80k_voc2012.py \
work_dirs/assignment_segmenter_vit-b16_80k_voc2012/latest.pth \
4 \
--format-only \
--eval-options "imgfile_prefix=work_dirs/assignment_segmenter_vit-b16_80k_voc2012/predictions"


# sudo bash local.sh \
# configs/segmenter/assignment_segmenter_vit-b16_80k_voc2012.py

# sudo bash local.sh \
# configs/segmenter/assignment_fcn_swin_nano_320k_voc2012.py

# sudo bash local.sh \
# configs/segmenter/assignment_segmenter_swin_nano_320k_voc2012.py
# configs/segmenter/assignment_segmenter_swin_nano_40k_voc2012.py


# python -m torch.distributed.launch --nproc_per_node=4 --master_port=233333 \
# tools/train.py \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs_16_downsample_2_base_config_structure_contrastive_loss_weight_1.0_coco-stuff_local.py \
# --launcher pytorch 

# python -m torch.distributed.launch --nproc_per_node=4 --master_port=233333 \
# tools/train.py \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs_16_downsample_2_base_config_structure_margin_0.3_min_0.0_loss_weight_1.0_coco-stuff_local.py \
# --launcher pytorch 


# sleep 10

# python -m torch.distributed.launch --nproc_per_node=4 --master_port=233333 \
# tools/train.py \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs_16_downsample_2_base_config_structure_cos_loss_weight_10.0_local.py \
# --launcher pytorch

# sleep 10

# python -m torch.distributed.launch --nproc_per_node=4 --master_port=233333 \
# tools/train.py \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_downsample_2_coco-stuff164k_PCA_eval_on_ade.py \
# --launcher pytorch


# python -m torch.distributed.launch \
# --nproc_per_node=4 --master_port=233333 \
# tools/test.py \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_eval_on_ade.py \
# work_dirs/segmenter-propagate_vit-b16_512x512_320k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_v2_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local/iter_224000.pth \
# --eval mIoU \
# --launcher pytorch

# python -m torch.distributed.launch \
# --nproc_per_node=4 --master_port=233333 \
# tools/test.py \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_eval_on_ade.py \
# work_dirs/segmenter-propagate_vit-b16_512x512_40k_bs16_downsample_2_coco-stuff164k_PCA_eval_on_ade/iter_40000.pth \
# --eval mIoU \
# --launcher pytorch



# configs/segmenter/segmenter-propagate_vit-b16_512x512_320k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_v2_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local.py \
# --auto-resume

# sudo bash local_test.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_eval_on_ade.py \
# work_dirs/segmenter-propagate_vit-b16_512x512_320k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_v2_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local/iter_224000.pth \
# 4 \
# --eval mIoU

# mv -v ../OpenVocSeg/pretrained_models/cls_emb_in21k_all_vild_split1.pth   pretrain/cls_emb_in21k_all_vild_split1.pth
# mv -v ../OpenVocSeg/pretrained_models/cls_emb_in21k_all_vild_split2.pth pretrain/cls_emb_in21k_all_vild_split2.pth                                                             
# mv -v ../OpenVocSeg/pretrained_models/cls_emb_in21k_all_vild_split3.pth  pretrain/cls_emb_in21k_all_vild_split3.pth                                                      
# mv -v ../OpenVocSeg/pretrained_models/cls_emb_in21k_all_vild_split4.pth  pretrain/cls_emb_in21k_all_vild_split4.pth


# sudo bash local_test.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_eval_on_ade_full.py \
# work_dirs/segmenter-propagate_vit-b16_512x512_320k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_v2_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local/iter_320000.pth \
# 4 \
# --format-only \
# --eval-options "imgfile_prefix=work_dirs/segmenter-propagate_vit-b16_512x512_320k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_v2_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local/inference_imagenet"
