_base_ = [
    '../_base_/datasets/wildscenes.py', '../_base_/models/spvcnn.py',
    '../_base_/default_runtime.py'
]

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=3, use_dim=3),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='wildscenes'),
    dict(type='PointSegClassMapping'),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0., 6.28318531],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
    ),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask', 'pts_hist'])
]

train_dataloader = dict(
    sampler=dict(seed=0), dataset=dict(pipeline=train_pipeline))

lr = 0.24
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='SGD', lr=lr, weight_decay=0.0001, momentum=0.9, nesterov=True))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.008, by_epoch=False, begin=0, end=125),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=50,
        by_epoch=True,
        eta_min=1e-5,
        convert_to_iter_based=True)
]

model = dict(
    data_preprocessor=dict(
        max_voxels=80000,
        voxel_layer=dict(
            voxel_size=[0.05,0.05,0.05],
        )
    ),
    decode_head = dict(
        ignore_index=255,
        num_classes=13,
        loss_decode=dict(
            type='mmdet.CrossEntropyLoss',
            avg_non_ignore=True,
            class_weight=None,
            ignore_index=255
        ))

)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1,
                                     save_best='miou', max_keep_ckpts=2))
env_cfg = dict(cudnn_benchmark=True)
