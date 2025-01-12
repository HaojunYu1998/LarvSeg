# MixBatch dataset
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
# Pascal Context
pascal_data_root = "/workspace/dataset/pcontext_full"
pascal_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=False),
    dict(type="Resize", img_scale=(520, 520), ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=(480, 480), cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=(480, 480), pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
pascal_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(520, 520),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ])
]
# ADE20K dataset settings
ade_data_root = "/workspace/dataset/ADEChallengeData2016"
ade_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=True),
    dict(type="Resize", img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type="MixBatchDataset",
        dataset_list=[
            dict(
                type="PascalContextDataset459",
                data_root=pascal_data_root,
                img_dir="train/image",
                ann_dir="train/label",
                pipeline=pascal_train_pipeline,
            ),
            dict(
                type="ADE20KDataset",
                data_root=ade_data_root,
                img_dir="images/training",
                ann_dir="annotations/training",
                pipeline=ade_train_pipeline,
            ),
        ],
    ),
    val=dict(
        type="PascalContextDataset459",
        data_root=pascal_data_root,
        img_dir="val/image",
        ann_dir="val/label",
        pipeline=pascal_test_pipeline,
    ),
    test=dict(
        type="PascalContextDataset459",
        data_root=pascal_data_root,
        img_dir="val/image",
        ann_dir="val/label",
        pipeline=pascal_test_pipeline,
    ),
)