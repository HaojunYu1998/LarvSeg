export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# bash tools/dist_train.sh \
# configs/large_voc/vit/large_voc_vitb16_cosine_80k_bs16_ade150_prior_loss.py

# bash tools/dist_train.sh \
# configs/large_voc/vit/large_voc_vitb16_cosine_80k_bs16_ade150_prior_structure_loss.py

# bash tools/dist_train.sh \
# configs/large_voc/vit/large_voc_vitb16_cosine_80k_bs16_ade150_structure_loss.py

# bash tools/dist_train.sh \
# configs/large_voc/vit/large_voc_vitb16_cosine_80k_bs16_ade847_prior_loss.py

# bash tools/dist_train.sh \
# configs/large_voc/vit/large_voc_vitb16_cosine_80k_bs16_ade847_prior_structure_loss.py

# bash tools/dist_train.sh \
# configs/large_voc/vit/large_voc_vitb16_cosine_80k_bs16_ade847_structure_loss.py

# bash tools/dist_train.sh \
# configs/large_voc/vit/large_voc_vitb16_linear_80k_bs16_cocostuff_prior_structure_loss.py

# bash tools/dist_train.sh \
# configs/large_voc/vit/large_voc_vitb16_linear_80k_bs16_cocostuff_prior_loss.py

# bash tools/dist_train.sh \
# configs/large_voc/vit/large_voc_vitb16_linear_80k_bs16_ade847_prior_structure_loss.py

# bash tools/dist_train.sh \
# configs/large_voc/vit/large_voc_vitb16_linear_80k_bs16_ade847_prior_loss.py

bash tools/dist_train.sh \
configs/large_voc/vit/large_voc_vitb16_linear_80k_bs16_ade150_prior_structure_loss.py

# bash tools/dist_train.sh \
# configs/large_voc/vit/large_voc_vitb16_linear_80k_bs16_ade150_prior_loss.py