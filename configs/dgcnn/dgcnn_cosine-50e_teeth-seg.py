_base_ = [
    '../_base_/datasets/teeth-seg.py', '../_base_/models/dgcnn.py',
    '../_base_/schedules/seg-cosine-50e.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    backbone=dict(in_channels=15),
    decode_head=dict(
        num_classes=17, ignore_index=17,
        loss_decode=dict(class_weight=None)
    ),
    test_cfg=dict(mode='whole')
)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=2))
train_dataloader = dict(batch_size=4)
train_cfg = dict(val_interval=1)