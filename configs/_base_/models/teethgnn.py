# model settings
model = dict(
    type='EncoderDecoder3DBranch',
    #NOTE 需要更改preprocessor设置, to enable 'adjacency_matrix' in 'inputs'
    data_preprocessor=dict(type='Det3DDataPreprocessorPlus'),
    backbone=dict(
        type='StaticGCNNBackbone',
        in_channels=15,
        knn_modes=('Adjacent', 'Adjacent', 'Adjacent'),
        gf_channels=((64, 64), (64, 64), (64, )),
        fa_channels=(1024, ),
        # TODO add gf_pool_modes & fa_pool_mode
        act_cfg=dict(type='LeakyReLU', negative_slope=0.2)
    ),
    decode_head=dict(
        type='TwoBranchHead',
        sem_channels=(1216, 256), 
        off_channels=(1216, 256, 128),
        co_shift=0.17,
        knn_modes=('Guided', ),
        gf_channels=((256, ), ),
        fp_channels=(256, 256),
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        # TODO default is 'ReLU' and previous TeethGNN use which kind?
        act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
        loss_semantic=dict(
            type='mmdet.CrossEntropyLoss',  #NOTE 需要加上mmdet.!
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0,
            avg_non_ignore=True),
        loss_offset_norm=dict(
            type='mmdet.L1Loss',  #NOTE 需要加上mmdet.!
            reduction='mean',
            loss_weight=3.0),
        use_direction_loss=False,
        zero_offset_classes=(0, 16),
        channels=128,
        dropout_ratio=0
    ),
    #model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide')
)