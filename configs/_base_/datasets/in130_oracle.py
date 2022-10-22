# MixBatch dataset
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# ImageNet21K dataset settings
in21k_data_root = '/mnt/haojun2/dataset/imagenet22k_azcopy'
in21k_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(320, 320), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(320, 320), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(320, 320), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
in21k_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='ImageNet130',
        data_root=in21k_data_root,
        img_dir='fall11_whole',
        ann_dir='annotations_cam_new',
        split="ImageNet130_vis20.txt",
        img_suffix=".JPEG",
        pipeline=in21k_train_pipeline),
    val=dict(
        type='ImageNet130',
        data_root=in21k_data_root,
        oracle_inference=True,
        img_dir='fall11_whole',
        ann_dir='annotations_cam_new',
        split="ImageNet130_vis20.txt",
        img_suffix=".JPEG",
        pipeline=in21k_test_pipeline),
    test=dict(
        type='ImageNet130',
        data_root=in21k_data_root,
        oracle_inference=True,
        img_dir='fall11_whole',
        ann_dir='annotations_cam_new',
        split="ImageNet130_vis20.txt",
        img_suffix=".JPEG",
        pipeline=in21k_test_pipeline),)