from os import path as osp
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

from mmdet3d.registry import DATASETS
from .seg3d_dataset import Seg3DDataset


@DATASETS.register_module()
class TeethSegDataset(Seg3DDataset):
    r"""Introduction...

    """
    METAINFO = {
        'classes':
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
        'palette': [[174, 199, 232],[152, 223, 138],[31, 119, 180],
            [255, 187, 120],[188, 189, 34],[140, 86, 75],
            [255, 152, 150],[214, 39, 40],[197, 176, 213],
            [148, 103, 189],[196, 156, 148],[23, 190, 207],
            [247, 182, 210],[219, 219, 141],[255, 127, 14],
            [158, 218, 229],[44, 160, 44],],
        'seg_valid_class_ids':
        tuple(range(17)),
        'seg_all_class_ids':
        tuple(range(17))
    }

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='points', pts_instance_mask='', pts_semantic_mask=''
                 ),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 ignore_index: Optional[int] = None,
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
            test_mode=test_mode,
            **kwargs
        )
    