class_names = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
metainfo = dict(classes=class_names)
dataset_type = 'TeethSegDataset'
data_root = 'data/teeth/10000x15'
input_modality = dict(use_lidar=True, use_camera=False)
data_prefix = dict(
    pts='points',
    pts_semantic_mask='semantic_mask',
    # adjacency_matrix='adjacency_matrix',
)
backend_args = None

num_points = 10000
# train_area = ... # not used here
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=15,
        use_dim=15,
        # backend_args=backend_args, # is this needed?
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        # with_adj_mat=True,
        # backend_args=backend_args, # is this needed?
    ),
    dict(type='PointSegClassMapping'),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask']) # 'adjacency_matrix'
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=15,
        use_dim=15,
        # backend_args=backend_args, # is this needed?
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        # with_adj_mat=True,
        # backend_args=backend_args, # is this needed?
    ),
    dict(type='Pack3DDetInputs', keys=['points']) # why 'pts_semantic_mask' is not needed?
]
eval_pipeline=[
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=15,
        use_dim=15,
        # backend_args=backend_args, # is this needed?
    ),
    dict(type='Pack3DDetInputs', keys=['points'])
]
tta_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=15,
        use_dim=15,
        # backend_args=backend_args, # is this needed?
    ),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        # with_adj_mat=True,
        # backend_args=backend_args, # is this needed?
    ),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=0.
            )],
            [dict(type='Pack3DDetInputs', keys=['points'])]
        ]
    )
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='teeth_infos_train.pkl',
        metainfo=metainfo,
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        modality=input_modality,
        ignore_index=len(class_names),
        test_mode=False,
        # backend_args=backend_args, # is this needed?
    )
)
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='teeth_infos_test_10.pkl',
        metainfo=metainfo,
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        modality=input_modality,
        ignore_index=len(class_names),
        test_mode=True,
        # backend_args=backend_args, # is this needed?
    )
)
val_dataloader = test_dataloader

val_evaluator = dict(type='SegMetric')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)

tta_model = dict(type='Seg3DTTAModel')
