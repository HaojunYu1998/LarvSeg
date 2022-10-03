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
    decode_head=dict(
        type="MaskTransformerLSegHead",
        input_transform="multiple_select",
        in_index=[0, 1, 2, 3],
        n_cls=171,
        downsample_rate=2,
        prior_rate=1.0,
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

# log_config = dict( 
#     interval=50, 
#     hooks=[ 
#         dict(type='TextLoggerHook'), 
#         dict(type='WandbLoggerHook', 
#             init_kwargs=dict(
#                 id="202209228_baseline_no_attn_cosine_160k_bs16_coco", 
#                 name="202209228_baseline_no_attn_cosine_160k_bs16_coco", 
#                 entity='haojunyu',
#                 project='SVLSeg',
#         ))
# ])