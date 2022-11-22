_base_ = [
    "../../_base_/models/segmenter_vit-b16.py",
    "../../_base_/datasets/ade_oracle.py",
    "../../_base_/default_runtime.py",
    "../../_base_/schedules/schedule_160k.py",
]

model = dict(
    type="EncoderDecoderOracle",
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
        n_cls=150,
        downsample_rate=2,
        cls_emb_path="pretrain/cls_emb_ade_vild_v2.pth",
        cls_emb_path_test="pretrain/cls_emb_ade_vild_v2.pth",
        imagenet_prior_rate=0.05,
        imagenet_pseudo_label=False,
        prior_rate=1.0,
        imagenet_prior_loss_weight=0.05,
        propagation_loss_weight=0.0,
        grounding_inference=True,
        oracle_inference=True,
        num_oracle_points=1,
        oracle_downsample_rate=1,
    ),
    test_cfg=dict(mode="whole", crop_size=(512, 512), stride=(512, 512)),
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
#                 id="202209225_baseline_160k_bs16_ade_all_eval_ade_gmiou",
#                 name="202209225_baseline_160k_bs16_ade_all_eval_ade_gmiou",
#                 entity='haojunyu',
#                 project='SVLSeg',
#         ))
# ])
