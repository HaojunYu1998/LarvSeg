_base_ = [
    "../../../_base_/models/large_voc_vitb16.py",
    "../../../_base_/datasets/mix_batch_IN130_COCO171_ADE150W_eval_ADE150.py",
    "../../../_base_/default_runtime.py",
    "../../../_base_/schedules/schedule_160k.py",
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
        type="MaskTransformerLargeVocCoSegHead",
        n_cls=150,
        downsample_rate=2,
        # datasets
        all_cls_path="notebook/ade150ucoco.json",
        mix_batch_datasets=["in130", "coco171", "ade150"],
        test_dataset="ade150",
        ignore_indices=[255, 255, 255],
        test_ignore_index=255,
        # weakly supervised
        weakly_supervised_datasets=["in130", "ade150"],
        weakly_basic_loss_weight=0.2,
        weakly_seed_loss_weight=0.1,
        # memory bank
        memory_bank_size=20,
        memory_image_size=20,
        use_memory_bank=False,
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