_base_ = [
    '../_base_/datasets/wildscenes.py', '../_base_/models/cylinder3d.py',
    '../_base_/default_runtime.py'
]

# optimizer
lr = 0.001
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[30],
        gamma=0.1)
]

train_dataloader = dict(batch_size=10, ) # can only use a batch size of 10 for Cylinder3D

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=32)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, save_best='miou', max_keep_ckpts=2))

grid_shape = [480, 360, 58] # 100%
# grid_shape = [360, 270, 24] # 75%
# grid_shape = [240, 180, 16] # 75%


model = dict(
    type='Cylinder3D',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='cylindrical',
        voxel_layer=dict(
            grid_shape=grid_shape,
            point_cloud_range=[0, -3.14159265359, -4, 50, 3.14159265359, 10],
            max_num_points=-1,
            max_voxels=-1,
        ),
    ),
    voxel_encoder=dict(
        type='SegVFE',
        feat_channels=[64, 128, 256, 256],
        in_channels=5,
        with_voxel_center=True,
        feat_compression=16,
        return_point_feats=False),
    backbone=dict(
        type='Asymm3DSpconv',
        grid_size=grid_shape,
        input_channels=16,
        base_channels=32,
        norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1)),
    decode_head=dict(
        type='Cylinder3DHead',
        channels=128,
        num_classes=13,
        loss_ce=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0,
            ignore_index=255),
        loss_lovasz=dict(type='LovaszLoss', loss_weight=1.0, reduction='none'),
    ),
    train_cfg=None,
    test_cfg=dict(mode='whole'),
)
