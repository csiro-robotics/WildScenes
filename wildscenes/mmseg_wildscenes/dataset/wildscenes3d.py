# Copyright (c) OpenMMLab. All rights reserved.
# Acknowledgements to the mmdetection3d repository: https://github.com/open-mmlab/mmdetection3d
# Original file "semantickitt_dataset.py" modified for the WildScenes dataset.

from typing import Callable, List, Optional, Sequence, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.datasets.seg3d_dataset import Seg3DDataset


@DATASETS.register_module()
class WildScenesDataset3d(Seg3DDataset):
    r"""Based off the format of the SemanticKITTI dataset.

    This class serves as the API for experiments on the WildScenes Dataset

    Args:
        data_root (str, optional): Path of dataset root. Defaults to None.
        ann_file (str): Path of annotation file. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(pts='',
                 img='',
                 pts_instance_mask='',
                 pts_semantic_mask='').
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input,
            it usually has following keys:

                - use_camera: bool
                - use_lidar: bool
            Defaults to dict(use_lidar=True, use_camera=False).
        ignore_index (int, optional): The label index to be ignored, e.g.
            unannotated points. If None is given, set to len(self.classes) to
            be consistent with PointSegClassMapping function in pipeline.
            Defaults to None.
        scene_idxs (np.ndarray or str, optional): Precomputed index to load
            data. For scenes with many points, we may sample it several times.
            Defaults to None.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
    """
    METAINFO = {
        "classes": (
            "bush",  # 0
            "dirt",  # 1
            "fence",  # 2
            "grass",  # 3
            "gravel",  # 4
            "log",  # 5
            "mud",  # 6
            "object",  # 7
            "other-terrain",  # 8
            "rock",  # 9
            "structure",  # 10
            "tree-foliage",  # 11
            "tree-trunk",  # 12
        ),
        "palette": [
            (230, 25, 75),
            (60, 180, 75),
            (0, 128, 128),
            (128, 128, 128),
            (145, 30, 180),
            (128, 128, 0),
            (255, 225, 25),
            (250, 190, 190),
            (70, 240, 240),
            (170, 255, 195),
            (170, 110, 40),
            (210, 245, 60),
            (240, 50, 230)],
        'seg_valid_class_ids':
            tuple(range(13)),
        'seg_all_class_ids':
            tuple(range(13)),
    }

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='',
                     img='',
                     pts_instance_mask='',
                     pts_semantic_mask='',
                     hist=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 ignore_index: Optional[int] = 255,
                 scene_idxs: Optional[Union[str, np.ndarray]] = None,
                 test_mode: bool = False,
                 **kwargs) -> None:
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            ignore_index=ignore_index,
            scene_idxs=scene_idxs,
            test_mode=test_mode,
            **kwargs)

    def get_seg_label_mapping(self, metainfo):
        seg_label_mapping = np.zeros(metainfo['max_label'] + 1, dtype=np.int64)
        for idx in metainfo['seg_label_mapping']:
            seg_label_mapping[idx] = metainfo['seg_label_mapping'][idx]
        return seg_label_mapping
