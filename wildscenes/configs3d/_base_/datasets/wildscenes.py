# This file has been modified from the config file of semanticKITTI to suit WildScenes
# https://github.com/open-mmlab/mmdetection3d/blob/main/configs/_base_/datasets/semantickitti.py
# Attribution: mmdetection3d, Apache 2.0 license

# For labels_map we follow the uniform format of MMDetection & MMSegmentation
# i.e. we consider the unlabeled class as the last one, which is different
# from the original implementation of some methods e.g. Cylinder3D.
dataset_type = 'WildScenesDataset3d'

# Please modify the path below and replace FULLPATHTOTHISREPO with where you installed this repo
data_root = 'FULLPATHTOTHISREPO/WildScenes/data/processed/wildscenes_opt3d'

# Raw labels (For visualising how to change)
# labels_map = {
#     0: 255,  # "unlabelled"
#     1: 0,  # asphalt/concrete
#     2: 1,  # dirt
#     3: 2,  # mud
#     4: 3,  # water
#     5: 4,  # sand
#     6: 5,  # gravel
#     7: 6,  # otherterrain
#     8: 7,  # treetrunk
#     9: 8,  # tree-foliage
#     10: 9,  # bush/shrub
#     11: 10,  # fence
#     12: 11,  # building
#     13: 12,  # other-structure
#     14: 13,  # pole
#     15: 14,  # vehicle
#     16: 15,  # rock
#     17: 16,  # log
#     18: 17,  # other-object
#     19: 18,  # sky
#     20: 19,  # grass
# }

class_names = (
            "bush", # 0
            "dirt", # 1
            "fence", # 2
            "grass", # 3
            "gravel", # 4
            "log", # 5
            "mud", # 6
            "object", # 7
            "other-terrain", # 8
            "rock", # 9
            "structure", # 10
            "tree-foliage", # 11
            "tree-trunk", # 12
        )

labels_map = {
    255:255, # Unlabeled
    0:0, # Bush 
    1:1, # Dirt
    2:2, # Fence
    3:3, # Grass
    4:4, # Gravel
    5:5, # Log
    6:6, # Mud
    7:7, # Object
    8:8, # Other-Terrain
    9:9, # Rock
    10:255, # Sky (IGNORED) 
    11:10, # Structure 
    12:11, # Tree-Foliage
    13:12, # Tree-Trunk
    14: 255 # Water (IGNORED)
}

metainfo = dict(
    classes=class_names, seg_label_mapping=labels_map, max_label=255)

input_modality = dict(use_lidar=True, use_camera=False)

backend_args = None

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=3,
        use_dim=3,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='wildscenes',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask', 'pts_hist'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=3,
        use_dim=3,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='wildscenes',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask', 'pts_hist'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=3,
        use_dim=3,
        backend_args=backend_args),
    dict(type='Pack3DDetInputs', keys=['points'])
]
tta_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=3,
        use_dim=3,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='wildscenes',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=0.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=1.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=0.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=1.)
        ],
                    [
                        dict(
                            type='GlobalRotScaleTrans',
                            rot_range=[pcd_rotate_range, pcd_rotate_range],
                            scale_ratio_range=[
                                pcd_scale_factor, pcd_scale_factor
                            ],
                            translation_std=[0, 0, 0])
                        for pcd_rotate_range in [-0.78539816, 0.0, 0.78539816]
                        for pcd_scale_factor in [0.95, 1.0, 1.05]
                    ], [dict(type='Pack3DDetInputs', keys=['points'])]])
]

train_dataloader = dict(
    batch_size=20,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='wildscenes_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=255,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='wildscenes_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=255,
        test_mode=True,
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='wildscenes_infos_test.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=255,
        test_mode=True,
        backend_args=backend_args))

# val_dataloader = test_dataloader

val_evaluator = dict(type='SegMetric')
# test_evaluator = val_evaluator
test_evaluator = dict(type='SegMetric')

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

tta_model = dict(type='Seg3DTTAModel')
randomness = dict(seed=0, deterministic=False, diff_rank_seed=True)
