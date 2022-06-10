_base_ = [
    "../_base_/models/segmenter_swin-nano.py",
    "../_base_/datasets/pascal_voc.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_320k.py",
]

model = dict(
    backbone=dict(
        type="SwinTransformer",
        pretrain_img_size=224,
        embed_dims=48,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.0,
        patch_norm=True),
    neck=dict(
        type="UseIndexSingleOutNeck",
        index=-1,
    ),
    decode_head=dict(
        type="MaskTransformerHead",
        n_cls=21,
        cls_emb_from_backbone=False,
    ),
    test_cfg=dict(mode="slide", crop_size=(480, 480), stride=(480, 480)),
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
        }),
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
