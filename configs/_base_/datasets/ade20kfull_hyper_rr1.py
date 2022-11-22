# dataset settings
dataset_type = "ADE20KFULLHyperDataset"
data_root = "/mnt/haojun2/dataset/ADE20K_2021_17_01"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
crop_size = (512, 512)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotationsI16", reduce_zero_label=True),
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
    dict(
        type="MultiScaleFlipAug",
        img_scale=(
            2048,
            512,
        ),  # [(2048, 512), (3072, 768), (4096, 1024)], #, #(6144, 1536), #(7168, 1792), #[(2048, 512), (3072, 768), (4096, 1024)], #, (6144, 1536), ],
        # img_ratios=[1.0, 1.5, 2.0, 2.5, 3.0],
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images_detectron2/training",
        ann_dir="annotations_detectron2/training",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images_detectron2/validation",
        ann_dir="annotations_detectron2/validation_hypernym",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images_detectron2/validation",
        ann_dir="annotations_detectron2/validation_hypernym",
        pipeline=test_pipeline,
    ),
)
