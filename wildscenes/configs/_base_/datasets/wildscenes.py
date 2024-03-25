dataset_type = "WildscenesDataset"
data_root = "/home/hau047/PycharmProjects/WildScenes/data/processed/wildscenes_opt2d"
data_prefix=dict(img_path="test/image", seg_map_path="test/indexLabel")
crop_size = (512, 512)
num_classes = 18
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(2016, 1512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
train_dataset = dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/image', seg_map_path='train/indexLabel'),
        pipeline=train_pipeline)
train_dataloader = dict(
    batch_size=2, # correct value = 20
    num_workers=10, # correct value = 25
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=train_dataset)
val_dataset = dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/image',
            seg_map_path='val/indexLabel'),
        pipeline=test_pipeline)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset)
test_dataset = dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/image',
            seg_map_path='test/indexLabel'),
        pipeline=test_pipeline)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
