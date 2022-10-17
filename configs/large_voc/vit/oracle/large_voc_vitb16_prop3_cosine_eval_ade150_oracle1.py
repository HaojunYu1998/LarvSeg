_base_ = [
    "../../../_base_/models/large_voc_vitb16.py",
    "../../../_base_/datasets/ade_oracle.py",
    "../../../_base_/default_runtime.py",
    "../../../_base_/schedules/schedule_160k.py",
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
        type="MaskTransformerLargeVocPropagationHead",
        n_cls=150,
        downsample_rate=2,
        temperature=0.05,
        # datasets
        all_cls_path="",
        mix_batch_datasets=[],
        test_dataset="ade150",
        ignore_indices=[],
        test_ignore_index=255,
        # attention head
        d_encoder=768,
        n_layers=3,
        n_heads=12,
        d_model=768,
        d_ff=4 * 768,
        drop_path_rate=0.0,
        dropout=0.1,
        use_linear_classifier=False,
        # weakly supervised
        weakly_supervised_datasets=[],
        weakly_prior_thresh=0.9,
        weakly_min_kept=10,
        weakly_max_kept=1000,
        weakly_prior_loss_weight=0.05,
        # contrastive loss
        structure_loss_weight=1.0,
        structure_loss_thresh=0.0,
        # oracle experiment
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
