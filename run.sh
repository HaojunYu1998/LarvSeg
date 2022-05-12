# sudo bash local.sh \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_v2_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local.py
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_v2_prior_0.05_loss_weight_0.1_mix_batch_coco-stuff164k_imagenet21k_local.py
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_pseudo_label_cam_vild_v2_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local.py

# # sudo bash local_test.sh \
# # configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_in21k_ade_filter_vild_eval_on_in21k.py \
# # work_dirs/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local/iter_40000.pth \
# # 4 \
# # --format-only \
# # --eval-options "imgfile_prefix=work_dirs/segmenter-propagate_vit-b16_512x512_40k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local/inference_imagenet"

touch /mnt/data0/suspend.txt

export OMP_NUM_THREADS=1


# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_downsample_2_base_config_in21k_bases_1000_fold0_rr1.py

# python -m torch.distributed.launch --nproc_per_node=8 --master_port=233333 \
# tools/train.py \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_downsample_2_base_config_in21k_bases_500_fold0_rr1.py \
# --launcher pytorch


python -m torch.distributed.launch --nproc_per_node=8 --master_port=233333 \
tools/train.py \
configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs_16_downsample_2_base_config_structure_contrastive_loss_weight_1.0_coco-stuff_rr1.py \
--launcher pytorch

# python -m torch.distributed.launch --nproc_per_node=8 --master_port=233333 \
# tools/train.py \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_40k_bs16_base_config_structure_loss_rr1.py \
# --launcher pytorch

# /mnt/haojun2/azcopy_linux_amd64_10.14.1/azcopy copy "https://itpsea4data.blob.core.windows.net/v-miazhang/dataset/imagenet22k_azcopy/fall11_whole/?sv=2020-10-02&st=2022-05-10T07%3A58%3A10Z&se=2022-05-17T07%3A58%3A10Z&sr=c&sp=rl&sig=Td2CXqHJRueuMNUfdCIvno160LVT7HPLC0cik0kSY0w%3D" "https://resrchvc4data.blob.core.windows.net/v-miazhang/dataset/imagenet22k_azcopy/?sv=2020-10-02&se=2022-06-09T08%3A13%3A14Z&sr=c&sp=rwl&sig=whgeE06dQfWUqMbSqh7dGVYO3zTV3lL3ndO4T35xmy8%3D" --overwrite=prompt --s2s-preserve-access-tier=false --include-directory-stub=false --recursive --log-level=INFO;


# mv -v ../OpenVocSeg/pretrained_models/cls_emb_in11k_bases_1000_fold0.pth ./pretrain/
# mv -v ../OpenVocSeg/pretrained_models/cls_emb_in11k_bases_1000_fold1.pth ./pretrain/
# mv -v ../OpenVocSeg/pretrained_models/cls_emb_in11k_bases_1000_fold2.pth ./pretrain/
# mv -v ../OpenVocSeg/pretrained_models/cls_emb_in11k_bases_1000_fold3.pth ./pretrain/
# mv -v ../OpenVocSeg/pretrained_models/cls_emb_in11k_bases_1000_fold4.pth ./pretrain/

# mv -v ../OpenVocSeg/pretrained_models/cls_emb_in11k_bases_500_fold0.pth ./pretrain/
# mv -v ../OpenVocSeg/pretrained_models/cls_emb_in11k_bases_500_fold1.pth ./pretrain/
# mv -v ../OpenVocSeg/pretrained_models/cls_emb_in11k_bases_500_fold2.pth ./pretrain/
# mv -v ../OpenVocSeg/pretrained_models/cls_emb_in11k_bases_500_fold3.pth ./pretrain/
# mv -v ../OpenVocSeg/pretrained_models/cls_emb_in11k_bases_500_fold4.pth ./pretrain/


# python -m torch.distributed.launch \
# --nproc_per_node=8 --master_port=233333 \
# tools/test.py \
# configs/segmenter/segmenter-propagate_vit-b16_512x512_eval_on_ade_hyper.py \
# work_dirs/segmenter-propagate_vit-b16_512x512_320k_bs16_downsample_2_base_config_ade_supervised/iter_192000.pth \
# --eval mIoU \
# --launcher pytorch
