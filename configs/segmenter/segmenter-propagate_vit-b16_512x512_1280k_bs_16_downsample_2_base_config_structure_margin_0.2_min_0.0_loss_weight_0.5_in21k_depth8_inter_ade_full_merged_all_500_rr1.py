_base_ = [
    "../_base_/models/segmenter_vit-b16.py",
    "../_base_/datasets/mix_batch_coco-stuff164k_imagenet21k_depth8_ade_full_rr1_v2.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_1280k.py",
]

model = dict(
    backbone=dict(
        drop_path_rate=0.1,
        final_norm=True,
    ),
    neck=dict(
        type="UseIndexSingleOutNeck",
        index=-1,
    ),
    decode_head=dict(
        type="MaskTransformerStructureHead",
        n_cls=655,
        downsample_rate=2,
        cls_emb_path=[
            "pretrain/cls_emb_coco_vild_v2.pth",
            "pretrain/cls_emb_in11k_depth8_dict.pth"
        ],
        cls_emb_path_test = "pretrain/cls_emb_ade_full_merged_def.pth",
        imagenet_class_path=None,
        imagenet_prior_rate=0.05,
        imagenet_pseudo_label=False,
        imagenet_sample_class_num=50,
        prior_rate=1.0,
        imagenet_prior_loss_weight=0.05,
        propagation_loss_weight=0.0,
        grounding_inference=True,
        ann_suffix=".tif",
        test_anno_dir="/mnt/haojun/dataset/ADE20K_2021_17_01/annotations_detectron2/validation_merged",
        structure_loss_method="margin",
        structure_loss_weight=0.5,
        structure_margin=0.2,
        structure_min_value=0.0
    ),
    test_cfg=dict(mode="slide", crop_size=(512, 512), stride=(512, 512)),
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
data = dict(samples_per_gpu=2)
