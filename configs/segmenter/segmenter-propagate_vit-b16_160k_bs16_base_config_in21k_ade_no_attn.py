_base_ = [
    "../_base_/models/segmenter_vit-b16.py",
    "../_base_/datasets/mix_batch_coco-stuff164k_imagenet21k_ade_filter_v2_rr1.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_160k.py",
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
        type="MaskTransformerPropagationHead",
        n_cls=150,
        downsample_rate=2,
        cls_emb_path=[
            "pretrain/cls_emb_coco_vild_v2.pth", # checked
            "pretrain/cls_emb_in21k_vild_v2.pth" # checked
        ],
        cls_emb_path_test="pretrain/cls_emb_ade_vild_v2.pth", # checked
        imagenet_class_path="notebook/in21k_inter_ade_filter_v2.json", # checked
        use_attention_module=False,
        imagenet_prior_rate=0.05,
        prior_rate=1.0,
        imagenet_prior_loss_weight=0.05,
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

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
