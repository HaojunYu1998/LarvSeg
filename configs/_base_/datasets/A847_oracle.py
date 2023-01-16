# data_root = "/workspace/dataset/A847"
data_root = "/workspace/dataset/ade20k_full"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (512, 512)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=False, int16=True),
    dict(type="Resize", img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=crop_size, pad_val=0, seg_pad_val=-1),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", reduce_zero_label=False, int16=True),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            # dict(type='ImageToTensor', keys=['img']),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_semantic_seg"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type="ADE20KFULLDataset",
        data_root=data_root,
        # img_dir="images/training",
        # ann_dir="annotations/training",
        img_dir="train/image",
        ann_dir="train/label",
        pipeline=train_pipeline,
    ),
    val=dict(
        type="ADE20KFULLDataset",
        oracle_inference=True,
        data_root=data_root,
        # img_dir="images/validation",
        # ann_dir="annotations/validation",
        img_dir="val/image",
        ann_dir="val/label",
        pipeline=test_pipeline,
    ),
    test=dict(
        type="ADE20KFULLDataset",
        oracle_inference=True,
        data_root=data_root,
        # img_dir="images/validation",
        # ann_dir="annotations/validation",
        img_dir="val/image",
        ann_dir="val/label",
        pipeline=test_pipeline,
    ),
)
