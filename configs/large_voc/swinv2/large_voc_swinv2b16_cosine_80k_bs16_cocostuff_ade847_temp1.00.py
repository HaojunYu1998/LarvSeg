_base_ = [
    "../../_base_/models/large_voc_swinv2b16.py",
    "../../_base_/datasets/mix_batch_cocostuff_adefull.py",
    "../../_base_/default_runtime.py",
    "../../_base_/schedules/schedule_80k.py",
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
        type="MaskTransformerLargeVocHead",
        n_cls=847,
        downsample_rate=2,
        temperature=1.00,
        # datasets
        all_cls_path="notebook/ade847ucoco.json",
        mix_batch_datasets=["coco171", "ade847"],
        test_dataset="ade847",
        # 65535 is -1 for int16, but during training the label will be cast to int32
        ignore_indices=[255, -1],
        test_ignore_index=-1,
        # weakly supervised
        weakly_supervised_datasets=["ade847"],
        weakly_prior_thresh=0.9,
        weakly_min_kept=10,
        weakly_max_kept=1000,
        # contrastive loss
        use_structure_loss=False,
        structure_loss_weight=1.0,
        structure_loss_thresh=0.0,
        # oracle experiment
        oracle_inference=False,
        num_oracle_points=10,
        oracle_downsample_rate=1,
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