from typing import Sequence, Union

from mmengine.model import BaseModule
from torch import Tensor
from torch import nn as nn

from mmdet3d.models.layers import StaticGCNNFAModule, StaticGCNNGFModule
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptMultiConfig


@MODELS.register_module()
class StaticGCNNBackbone(BaseModule):
    """Backbone network for StaticGCNN.

    Args:
        in_channels (int): Input channels of point cloud.
        knn_modes (tuple[str], optional): Mode of KNN of each knn module.
            Defaults to ('Adjacent', 'Adjacent', 'Adjacent').
        gf_channels (tuple[tuple[int]], optional): Out channels of each mlp in
            GF module. Defaults to ((64, 64), (64, 64), (64, )).
        fa_channels (tuple[int], optional): Out channels of each mlp in FA
            module. Defaults to (1024, ).
        act_cfg (dict, optional): Config of activation layer.
            Defaults to dict(type='ReLU').
        init_cfg (dict, optional): Initialization config.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 # use fixed number of adjacent neighbors (pre-calculated)
                 knn_modes: Sequence[str] = ('Adjacent', 'Adjacent', 'Adjacent'),
                 gf_channels: Sequence[Sequence[int]] = ((64, 64), (64, 64),
                                                         (64, )),
                 fa_channels: Sequence[int] = (1024, ),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.num_gf = len(gf_channels)

        self.GF_modules = nn.ModuleList()
        gf_in_channel = in_channels * 2
        skip_channel_list = [gf_in_channel]  # input channel list

        for gf_index in range(self.num_gf):
            cur_gf_mlps = list(gf_channels[gf_index])
            cur_gf_mlps = [gf_in_channel] + cur_gf_mlps
            gf_out_channel = cur_gf_mlps[-1]
        
            self.GF_modules.append(
                StaticGCNNGFModule(
                    mlp_channels=cur_gf_mlps,
                    knn_mode=knn_modes[gf_index],
                    act_cfg=act_cfg
                )
            )
            skip_channel_list.append(gf_out_channel)
            gf_in_channel = gf_out_channel * 2

        fa_in_channel = sum(skip_channel_list[1:])
        cur_fa_mlps = list(fa_channels)
        cur_fa_mlps = [fa_in_channel] + cur_fa_mlps

        self.FA_module = StaticGCNNFAModule(
            mlp_channels=cur_fa_mlps, act_cfg=act_cfg
        )

    def forward(self, points: Tensor, adjacency_matrix: Tensor) -> dict:
        """Forward pass.

        Args:
            points (torch.Tensor): Point coordinates with features,
                with shape [B, N, in_channels].
            adjacency_matrix (torch.Tensor): Adjacency matrix with shape
                [B, N, 3].

        Returns:
            dict[str, list[torch.Tensor]]: Outputs after graph feature (GF) and
                feature aggregation (FA) modules.

                - gf_points (list[torch.Tensor]): Outputs after each GF module.
                - fa_points (torch.Tensor): Outputs after FA module.
        """

        gf_points = [points]

        for i in range(self.num_gf):
            cur_points = self.GF_modules[i](
                gf_points[i],
                adjacency_matrix=adjacency_matrix
            )
            gf_points.append(cur_points)
        
        fa_points = self.FA_module(gf_points)

        out = dict(gf_points=gf_points, fa_points=fa_points)
        return out
