_base_ = [
    '../_base_/datasets/teeth-seg.py', '../_base_/models/teethgnn.py',
    '../_base_/schedules/seg-cosine-50e.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    backbone=dict(
        in_channels=15,
        knn_modes=('Adjacent', 'Adjacent', 'Adjacent'),
        # gf_pool_modes=('max', 'max', 'max'),
        # fa_pool_mode='avg'
    ),
    decode_head=dict(
        num_classes=17,
        ignore_index=17,
        co_shift=0.17,
        knn_modes=('Adjacent', ),
        dropout_ratio=0,
        loss_offset_norm=dict(loss_weight=3.0),
        use_direction_loss=False,
        zero_offset_classes=(0, 16)
    ),
    test_cfg=dict(mode='whole')
)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=2))
train_dataloader = dict(batch_size=4)
train_cfg = dict(val_interval=1)