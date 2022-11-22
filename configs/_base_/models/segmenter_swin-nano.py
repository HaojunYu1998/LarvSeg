# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
backbone_norm_cfg = dict(type="LN", requires_grad=True)
model = dict(
    type="EncoderDecoder",
    pretrained="pretrain/swin_nano_patch4_window7_224.pth",
    backbone=dict(
        type="SwinTransformer",
        pretrain_img_size=224,
        embed_dims=48,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(3,),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        use_abs_pos_embed=False,
        act_cfg=dict(type="GELU"),
        norm_cfg=backbone_norm_cfg,
    ),
    decode_head=dict(
        type="MaskTransformerHead",
        n_cls=150,
        patch_size=16,
        d_encoder=384,
        n_layers=2,
        n_heads=12,
        d_model=384,
        d_ff=4 * 384,
        drop_path_rate=0.0,
        dropout=0.1,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)  # yapf: disable
