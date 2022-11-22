_base_ = [
    "../_base_/models/large_voc_vitb16.py",
    "../_base_/datasets/mix_batch_WA847_C171_I585_eval_A847.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_320k.py",
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
        type="MaskTransformerExtendVocHead",
        n_cls=847,
        downsample_rate=2,
        all_cls_path="file/ade847ucoco.json",
        mix_batch_datasets=["ade847", "coco171", "in585"],
        weakly_supervised_datasets=["in585", "ade847"],
        test_dataset="ade847",
        ignore_indices=[-1, 255, -1],
        test_ignore_index=-1,
        use_sample_class=True,
        num_smaple_class=100,
        basic_loss_weights=[0.2, 1.0, 0.2],
        coseg_loss_weights=[0.1, 0.0, 0.1],
        use_coseg=True,
        use_coseg_score_head=False,
        memory_bank_size=10,
        memory_bank_warm_up=1,
        foreground_topk=20,
        background_suppression=True,
        background_topk=5,
        background_thresh=0.35,
        background_mse_thresh=2.0,
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
