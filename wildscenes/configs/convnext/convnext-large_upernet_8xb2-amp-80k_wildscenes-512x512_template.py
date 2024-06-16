_base_ = [
    '../_base_/models/upernet_convnext.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth'  # noqa
backbone = dict(
        type='mmpretrain.ConvNeXt',
        arch='large',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.'))
decode_head = dict(
        in_channels=[192, 384, 768, 1536],
    )
aux_head = dict(in_channels=768)
# test_cfg=dict(mode='slide', stride=(426, 426))
test_cfg = dict(type='TestLoop')
model = dict(
    backbone=backbone,
    decode_head=decode_head,
    auxiliary_head=aux_head,
    test_cfg=test_cfg
)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12
    },
    constructor='LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic')

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=80000,
        eta_min=0.0,
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
