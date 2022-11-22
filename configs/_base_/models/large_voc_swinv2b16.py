# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    type="EncoderDecoder",
    backbone=dict(
        type="SwinTransformerV2",
        img_size=512,
        num_classes=0,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        drop_path_rate=0.2,
        window_size=16,
        pretrained_window_sizes=[12, 12, 12, 6],
        pretrained="pretrain/swinv2_base_patch4_window12_192_22k.pth",
    ),
    decode_head=dict(
        type="MaskTransformerHead",
        n_cls=150,
        patch_size=16,
        d_encoder=1024,
        n_layers=2,
        n_heads=12,
        d_model=512,
        upsample_input=2,
        d_ff=4 * 512,
        drop_path_rate=0.0,
        dropout=0.1,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)  # yapf: disable
