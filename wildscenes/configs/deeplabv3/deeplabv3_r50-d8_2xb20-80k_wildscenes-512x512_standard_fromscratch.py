_base_ = [
    '../_base_/models/deeplabv3_r50-d8_nopretrain.py', '../_base_/datasets/wildscenes_standard.py',
    '../_base_/default_runtime.py', "../_base_/schedules/schedule_80k.py"
]

# Adapt the model and preprocessor to the crop size and number of classes
crop_size = _base_.crop_size
num_classes = _base_.num_classes
data_preprocessor = dict(size=crop_size, test_cfg=dict(size_divisor=32))
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=num_classes),
    auxiliary_head=dict(num_classes=num_classes))
# Add hooks for checkpointing and visualization
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint_hook = dict(type="CheckpointHook", by_epoch=False, interval=1000, max_keep_ckpts=2, save_best="mIoU"),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

