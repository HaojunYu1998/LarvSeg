_base_ = [
    # "./training_scheme.py",
    "../_base_/models/segmenter_vit-b16.py",
    # "../_base_/datasets/ade20kfull.py",
    "../_base_/datasets/mix_batch_coco-stuff164k_imagenet21k_ade_filter_v2_sea.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]

model = dict(
    type="EncoderDecoderV2",
    backbone=dict(
        drop_path_rate=0.1,
        final_norm=True,
    ),
    neck=dict(
        type="UseIndexSingleOutNeck",
        index=-1,
    ),
    decode_head=dict(
        type="MaskTransformerPropagationHead",
        # n_cls=847,
        # cls_emb_path="pretrain/cls_emb_ade_full_vild_v2.pth",
        # cls_emb_path_test="pretrain/cls_emb_ade_full_vild_v2.pth",
        n_cls=150,
        cls_emb_path="pretrain/cls_emb_ade_vild_v2.pth",
        cls_emb_path_test="pretrain/cls_emb_ade_vild_v2.pth",
        imagenet_class_path="notebook/in21k_inter_ade_filter_v2.json",
        # cls_emb_hyper_path_test="pretrain/cls_emb_ade_hyper_vild.pth",
        # cls_emb_hyper_ensemble_weight=0.0,
        # test_anno_dir="/data/ade20k_full/annotations_detectron2/validation_merged",
        prior_rate=0.1,
        # contrastive_propagation=True,
        propagation_loss_weight=1.0,
        # imagenet_pred_save_dir="work_dirs/segmenter-propagate_vit-b16_512x512_320k_bs16_prior_1.0_lambda_0.0_downsample_2_in21k_ade_filter_vild_v2_prior_0.05_loss_weight_0.05_mix_batch_coco-stuff164k_imagenet21k_local/in21k_mask/",
        # propagation_loss_mode="kl_div",
        grounding_inference=True,
        # ann_suffix=".tif",
        # ignore_index=65535
    ),
    test_cfg=dict(
        use_dense_crf_postprocess=True,
        mode="slide", 
        crop_size=(512, 512), 
        stride=(512, 512)
    ),
)

optimizer = dict(
    _delete_=True,
    type="SGD",
    lr=0.001,
    weight_decay=0.0,
    momentum=0.9,
    paramwise_cfg=dict(
        custom_keys={
            "pos_embed": dict(decay_mult=0.0),
            "cls_token": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

lr_config = dict(
    _delete_=True,
    policy="poly",
    warmup_iters=0,
    power=0.9,
    min_lr=1e-5,
    by_epoch=False,
)


# By default, models are trained on 8 GPUs with 1 images per GPU
data = dict(
    samples_per_gpu=4,
)
