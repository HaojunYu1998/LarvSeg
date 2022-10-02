_base_ = [
    "../_base_/models/segmenter_vit-b16.py",
    "../_base_/datasets/coco-stuff164k.py",
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
        use_attention_module=False,
        use_structure_loss=True,
        structure_loss_weight=10.0,
        n_cls=171,
        downsample_rate=2,
        cls_emb_path="pretrain/cls_emb_coco_vild_v2.pth",
        cls_emb_path_test="pretrain/cls_emb_coco_vild_v2.pth",
        prior_rate=1.0,
        propagation_loss_weight=0.0,
        grounding_inference=False,
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

# By default, models are trained on 4 GPUs with 4 images per GPU
data = dict(samples_per_gpu=4)
