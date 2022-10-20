_base_ = [
    "../../../_base_/models/large_voc_vitb16.py",
    "../../../_base_/datasets/mix_batch_IN130_COCO171_eval_ADE130.py",
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
        type="MaskTransformerLargeVocAttnHead",
        n_cls=130, # train on 256 classes, eval 130 classes
        downsample_rate=2,
        # datasets
        all_cls_path="notebook/ade130ucoco.json",
        mix_batch_datasets=["in130", "coco171"],
        test_dataset="ade130",
        ignore_indices=[255, 255],
        test_ignore_index=255,
        # attention head
        d_encoder=768,
        n_layers=2,
        n_heads=12,
        d_model=768,
        d_ff=4 * 768,
        drop_path_rate=0.0,
        dropout=0.1,
        # weakly supervised
        weakly_supervised_datasets=["in130"],
        weakly_seed_thresh=0.15,
        weakly_min_kept=1000,
        weakly_max_kept=25600,
        weakly_seed_loss_weight=0.2,
        # contrastive loss
        use_structure_loss=False,
        structure_loss_weight=1.0,
        structure_loss_thresh=0.0,
        # memory bank
        memory_bank_size=30,
        memory_bank_topk=50,
        memory_bank_weight=0.5,
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