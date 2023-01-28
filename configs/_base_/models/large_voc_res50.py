# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
d_model = 512
model = dict(
    type="EncoderDecoder",
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type="ResNetV1c",
        depth=50,
        style="pytorch",
        frozen_stages=-1,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        norm_eval=False,
    ),
    neck=dict(
        type="SingleOutFPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=d_model,
        num_outs=4,
        out_level=2,
        start_level=0,
        end_level=-1,
        add_extra_convs=False,
        extra_convs_on_inputs=False,
        relu_before_extra_convs=False,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        upsample_cfg=dict(mode="nearest"),
        init_cfg=dict(type="Xavier", layer="Conv2d", distribution="uniform"),
    ),
    decode_head=dict(
        type="LarvSegHead",
        n_cls=150,
        d_model=d_model,
        downsample_rate=1,
        all_cls_path="",
        ignore_cls_path="",
        mix_batch_datasets=["ade150"],
        weakly_supervised_datasets=[],
        test_dataset="ade150",
        ignore_indices=[255],
        test_ignore_index=255,
        basic_loss_weights=[1.0],
        coseg_loss_weights=[0.0],
        patch_size=16,
        d_encoder=256,
        n_layers=2,
        n_heads=12,
        d_ff=4 * 768,
        drop_path_rate=0.0,
        dropout=0.1,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)  # yapf: disable
