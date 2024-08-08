_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/wildscenes_standard.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
# Adjust model to our data and number of classes and load pretrained checkpoint
crop_size = _base_.crop_size
num_classes = _base_.num_classes
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
                  embed_dims=64,
                  num_layers=[3,6,40,3]),
    decode_head=dict(in_channels=[64, 128, 320, 512], 
                     num_classes=num_classes))
# Add the optimiser and scheduler that was used in cityscapes
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False,
    )
]
# Add hooks for checkpointing and visualization
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint_hook = dict(type="CheckpointHook", by_epoch=False, interval=1000, max_keep_ckpts=2, save_best="mIoU"),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
