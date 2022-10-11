_base_ = [
    "../../_base_/models/large_voc_vitb16.py",
    "../../_base_/datasets/mix_batch_COCO171_IN585_eval_ADE585.py",
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
        n_cls=585, # train on 686 classes, eval 585 classes
        downsample_rate=2,
        temperature=0.05,
        # datasets
        all_cls_path="notebook/ade585ucoco.json",
        mix_batch_datasets=["coco171", "in585"],
        test_dataset="ade585", # not used
        ignore_indices=[255, -1],
        test_ignore_index=-1, # used
        # attention head
        d_encoder=768,
        n_layers=6,
        n_heads=12,
        d_model=768,
        d_ff=4 * 768,
        drop_path_rate=0.0,
        dropout=0.1,
        # prior loss
        use_prior_loss=True,
        use_linear_classifier=False,
        # weakly supervised
        weakly_supervised_datasets=["in585"],
        weakly_prior_thresh=0.95,
        weakly_min_kept=10,
        weakly_max_kept=1000,
        weakly_prior_loss_weight=0.05,
        # contrastive loss
        use_structure_loss=False,
        structure_loss_weight=10.0,
        structure_loss_thresh=0.3,
        # oracle experiment
        oracle_inference=False,
        num_oracle_points=1,
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
