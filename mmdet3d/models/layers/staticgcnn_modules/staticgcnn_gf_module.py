from typing import List, Sequence, Optional, Union

import torch
from mmcv.cnn import ConvModule
from mmcv.ops.group_points import QueryAndGroup, grouping_operation
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.utils import ConfigType


class BaseStaticGCNNGFModule(nn.Module):
    """Base module for point graph feature module used in StaticGCNN.

    Args:
        mlp_channels (List[List[int]]): Specify of the dgcnn before the global
            pooling for each graph feature module.
        pool_mode (str): Type of pooling method. Defaults to 'max'.
    """

    def __init__(self, 
                 mlp_channels: List[List[int]],
                 knn_modes: List[str] = ['Adjacent',],
                 pool_mode: str = 'max') -> None:
        super(BaseStaticGCNNGFModule, self).__init__()

        assert pool_mode in ['max', 'avg'
                             ], "Pool_mode should be one of ['max', 'avg']."
        assert isinstance(knn_modes, list) or isinstance(
            knn_modes, tuple), 'The type of knn_modes should be list or tuple.'
        
        if isinstance(mlp_channels, tuple):
            mlp_channels = list(map(list, mlp_channels))
        self.mlp_channels = mlp_channels

        self.pool_mode = pool_mode
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.knn_modes = knn_modes

        for i in range(len(knn_modes)):
            knn_mode = self.knn_modes[i]
            if knn_mode in ['Guided']:
                grouper = QueryAndGroup(
                    None,                     # radius[i]
                    3,                        # num_samples[i]
                    use_xyz=True,             # use_xyz,
                    normalize_xyz=False,      # normalize_xyz,
                    return_grouped_xyz=False, # grouper_return_grouped_xyz,
                    return_grouped_idx=True)
            elif knn_mode in ['Adjacent']:
                grouper = None
            self.groupers.append(grouper)

    def _pool_features(self, features: Tensor) -> Tensor:
        """Perform feature aggregation using pooling operation.

        Args:
            features (Tensor): (B, C, N, K) Features of locally grouped
                points before pooling.

        Returns:
            Tensor: (B, C, N) Pooled features aggregating local information.
        """
        if self.pool_mode == 'max':
            # (B, C, N, 1)
            new_features = F.max_pool2d(
                features, kernel_size=[1, features.size(3)])
        elif self.pool_mode == 'avg':
            # (B, C, N, 1)
            new_features = F.avg_pool2d(
                features, kernel_size=[1, features.size(3)])
        else:
            raise NotImplementedError
        
        return new_features.squeeze(-1).contiguous()

    def forward(self, points: Tensor, guided_points: Optional[Tensor]=None, adjacency_matrix: Optional[Tensor]=None) -> Tensor:
        """forward.

        Args:
            points (Tensor): (B, N, C) Input points.
            adjacency_matrix (Tensor): (B, N, k) Precalculated neighbors 
                points idx.

        Returns:
            Tensor: (B, N, C1) New points generated from each graph
            feature module.
        """
        new_points_list = [points]

        for i in range(len(self.mlp_channels)):

            new_points = new_points_list[i]
            new_points_trans = new_points.transpose(
                1, 2).contiguous()  # (B, C, N)

            ### Guided-KNN
            if self.knn_modes[i] == 'Guided':
                assert guided_points is not None, \
                    "Guided-KNN must have guided points as input."
                # get knn based on guided points, idx: [B, N, k]
                idx = self.groupers[i](guided_points, guided_points)[-1]
                # [B, C, N] -> results: [B, C, N, K]
                grouped_results = grouping_operation(new_points_trans, idx)
                # features = features - x
                grouped_results -= new_points_trans.unsqueeze(-1)

            ### Adjacent-KNN
            elif self.knn_modes[i] == 'Adjacent':
                # adjacency_matrix: [B, N, k]
                assert adjacency_matrix is not None, \
                    "Adjacent-KNN must have adjacency matrix as input."
                # (B, C, N) -> (B, C, N, k)
                grouped_results = grouping_operation(
                    new_points_trans, 
                    adjacency_matrix)
                # features = features - x
                grouped_results -= new_points_trans.unsqueeze(-1)

            new_points = new_points_trans.unsqueeze(-1).repeat(
                1, 1, 1, grouped_results.shape[-1])
            new_points = torch.cat([grouped_results, new_points], dim=1)

            # (B, mlp[-1], N, K)
            new_points = self.mlps[i](new_points)

            # (B, mlp[-1], N)
            new_points = self._pool_features(new_points)
            new_points = new_points.transpose(1, 2).contiguous()
            new_points_list.append(new_points)
        
        return new_points
    

class StaticGCNNGFModule(BaseStaticGCNNGFModule):
    """Point graph feature module used in StaticGCNN.
    
    Args:
        mlp_channels (List[int]): Specify of the sgcnn before the global
            pooling for each graph feature module.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN2d').
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to dict(type='ReLU').
        pool_mode (str): Type of pooling method. Defaults to 'max'.
        bias (bool or str): If specified as `auto`, it will be decided by
            `norm_cfg`. `bias` will be set as True if `norm_cfg` is None,
            otherwise False. Defaults to 'auto'.
    """

    def __init__(self,
                 mlp_channels: List[int],
                 knn_mode: str = 'Adjacent',
                 norm_cfg: ConfigType = dict(type='BN2d'),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 pool_mode: str = 'max',
                 bias: Union[bool, str] = 'auto') -> None:
        super(StaticGCNNGFModule, self).__init__(
            mlp_channels=[mlp_channels],
            knn_modes=[knn_mode],
            pool_mode=pool_mode
        )

        for i in range(len(self.mlp_channels)):
            mlp_channel = self.mlp_channels[i]

            mlp = nn.Sequential()
            for i in range(len(mlp_channel) - 1):
                mlp.add_module(
                    f'layer{i}',
                    ConvModule(
                        mlp_channel[i],
                        mlp_channel[i+1],
                        kernel_size=(1,1),
                        stride=(1,1),
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        bias=bias
                    )
                )
            self.mlps.append(mlp)
