export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_LAUNCH_BLOCKING=1

# bash tools/dist_train.sh \
# configs/large_voc/vit/large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss.py

# bash tools/dist_test.sh \
# configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle1.py \
# work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
# 4 \
# --eval mIoU > structure1.txt

# bash tools/dist_test.sh \
# configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle3.py \
# work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
# 4 \
# --eval mIoU > structure3.txt

# bash tools/dist_test.sh \
# configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle10.py \
# work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
# 4 \
# --eval mIoU > structure10.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_prior1_v1.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_prior1_v2.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_prior1_v3.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_prior1_v4.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_prior1_v5.txt


bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_prior3_v1.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_prior3_v2.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_prior3_v3.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_prior3_v4.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_prior3_v5.txt


bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_prior10_v1.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_prior10_v2.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_prior10_v3.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_prior10_v4.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_prior10_v5.txt









bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_prior1_v1.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_prior1_v2.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_prior1_v3.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_prior1_v4.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_prior1_v5.txt


bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_prior3_v1.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_prior3_v2.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_prior3_v3.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_prior3_v4.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_prior3_v5.txt


bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_prior10_v1.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_prior10_v2.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_prior10_v3.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_prior10_v4.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_prior10_v5.txt








bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_prior1_v1.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_prior1_v2.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_prior1_v3.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_prior1_v4.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_prior1_v5.txt


bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_prior3_v1.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_prior3_v2.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_prior3_v3.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_prior3_v4.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_prior3_v5.txt


bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_prior10_v1.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_prior10_v2.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_prior10_v3.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_prior10_v4.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_prior_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_prior10_v5.txt














bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_structure1_v1.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_structure1_v2.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_structure1_v3.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_structure1_v4.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_structure1_v5.txt


bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_structure3_v1.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_structure3_v2.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_structure3_v3.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_structure3_v4.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_structure3_v5.txt


bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_structure10_v1.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_structure10_v2.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_structure10_v3.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_structure10_v4.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_coco171_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > coco171_structure10_v5.txt









bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_structure1_v1.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_structure1_v2.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_structure1_v3.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_structure1_v4.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_structure1_v5.txt


bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_structure3_v1.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_structure3_v2.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_structure3_v3.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_structure3_v4.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_structure3_v5.txt


bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_structure10_v1.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_structure10_v2.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_structure10_v3.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_structure10_v4.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade150_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade150_structure10_v5.txt








bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_structure1_v1.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_structure1_v2.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_structure1_v3.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_structure1_v4.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle1.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_structure1_v5.txt


bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_structure3_v1.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_structure3_v2.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_structure3_v3.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_structure3_v4.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle3.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_structure3_v5.txt


bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_structure10_v1.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_structure10_v2.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_structure10_v3.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_structure10_v4.txt

bash tools/dist_test.sh \
configs/large_voc/vit/oracle/large_voc_vitb16_cosine_eval_ade847_oracle10.py \
work_dirs/20221006_large_voc_vitb16_cosine_80k_bs16_cocostuff_structure_loss/iter_80000.pth \
4 \
--eval mIoU > ade847_structure10_v5.txt


# sudo nvidia-docker run --ipc=host -it -v /mnt/haojun/itpsea4data:/itpsea4data --ipc=host hsfzxjy/mmseg:pytorch1.8.1-cuda10.2-cudnn7-devel /bin/bash
# sudo nvidia-docker run --ipc=host -it -v /mnt/haojun/itpsea4data:/itpsea4data --ipc=host zeliu98/pytorch:superbench-nvcr21.05-fixfusedlamb-itp-mmcv-msrest /bin/bash

# pip install -U amlt --extra-index-url https://msrpypi.azurewebsites.net/stable/7e404de797f4e1eeca406c1739b00867 --extra-index-url https://azuremlsdktestpypi.azureedge.net/K8s-Compute/D58E86006C65
# apt update
# apt-get install unzip -y
# apt-get install htop -y
# apt-get install tmux -y
# apt-get install vim -y
# pip install install git+https://github.com/lucasb-eyer/pydensecrf.git
# cd third_party/CLIP
# pip install -e .
# cd ../detectron2
# pip install -e .
# cd ../..
# mkdir -p /mnt/haojun2
# ln -s /itpsea4data/dataset /mnt/haojun2/dataset
# pip install timm