from typing import Dict, List, Sequence

from mmcv.cnn.bricks import ConvModule
import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.models.layers import DGCNNFPModule, StaticGCNNGFModule
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils.typing_utils import ConfigType
from .decode_head import Base3DDecodeHead


@MODELS.register_module()
class TwoBranchHead(Base3DDecodeHead):
    r"""Two-branch decoder head.

    Decoder head used in `TeethGNN`.

    Args:
        fp_channels (Sequence[int]): 
    """

    def __init__(self, 
                 sem_channels: Sequence[int] = (1216, 256),
                 off_channels: Sequence[int] = (1216, 256, 128),
                 use_off_branch: bool =True,
                 co_shift: float = 1.0,
                 knn_modes: Sequence[str] = ('Guided', ),
                 gf_channels: Sequence[Sequence[int]] = ((256, ), ),
                 fp_channels: Sequence[int] = (256, 256),
                 loss_semantic: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0,
                     avg_non_ignore=True),
                 loss_offset_norm: ConfigType = dict(
                     type='mmdet.L1Loss',
                     reduction='mean',
                     loss_weight=3.0),  # strictly, needs to multiply 3 (3d-xyz)
                 use_direction_loss: bool = True,
                 zero_offset_classes: Sequence[int] = (0, ),
                 **kwargs):
        super(TwoBranchHead, self).__init__(**kwargs)

        self.use_off_branch = use_off_branch
        self.co_shift = co_shift
        self.num_gf = len(gf_channels)
        self.loss_semantic = MODELS.build(loss_semantic)
        self.loss_offset_norm = MODELS.build(loss_offset_norm)
        self.use_direction_loss = use_direction_loss
        self.zero_offset_classes = list(zero_offset_classes).sort()

        ### offset branch
        if self.use_off_branch:
            self.off_FP_module = DGCNNFPModule(mlp_channels=off_channels,
                                               act_cfg=self.act_cfg)
            self.off_conv = nn.Conv1d(off_channels[-1],
                                      out_channels=3,
                                      kernel_size=1,
                                      bias=False)
            
        ### semantic branch 
        # mlps
        self.sem_FP_module = DGCNNFPModule(mlp_channels=sem_channels, 
                                           act_cfg=self.act_cfg)
        
        # graph-features (edgeconv) modules
        self.GF_modules = nn.ModuleList()
        gf_in_channel = sem_channels[-1] * 2
        for gf_index in range(self.num_gf):
            cur_gf_mlps = list(gf_channels[gf_index])
            cur_gf_mlps = [gf_in_channel] + cur_gf_mlps
            gf_out_channel = cur_gf_mlps[-1]
            self.GF_modules.append(
                StaticGCNNGFModule(
                    mlp_channels=cur_gf_mlps,
                    knn_mode=knn_modes[gf_index],
                    act_cfg=self.act_cfg
                )
            )
            gf_in_channel = gf_out_channel * 2
        
        ### 2 branches joined
        self.join_FP_modules = DGCNNFPModule(mlp_channels=fp_channels,
                                             act_cfg=self.act_cfg)
        
        # pre-segmentation conv
        #  (after this, there's a dropout & the last conv)
        self.pre_seg_conv = ConvModule(
            fp_channels[-1],
            self.channels,  # what's this?
            kernel_size=1,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

    def _extract_input(self, feat_dict: dict) -> tuple:
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.
                - gf_points (list[torch.Tensor]): Outputs after each GF module.
                - fa_points (torch.Tensor): Outputs after FA module.
            

        Returns:
            tuple: Xyz coords and features of points.
        """
        xyz_points = feat_dict['gf_points'][0][..., :3].contiguous()
        fa_points = feat_dict['fa_points']

        return xyz_points, fa_points

    def forward(self, feat_dict: dict, adjacency_matrix: Tensor) -> dict:
        """Forward function for training."""
        if self.use_off_branch:
            return self.forward_2branch(feat_dict, adjacency_matrix)
        else:
            return self.forward_1branch(feat_dict, adjacency_matrix)
    
    def loss(self, inputs: dict, adjacency_matrix: Tensor, batch_data_samples: SampleList,
             train_cfg: ConfigType) -> Dict[str, Tensor]:
        """Forward function for training.

        Args:
            inputs (dict): Feature dict from backbone.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.
            train_cfg (dict or :obj:`ConfigDict`): The training config.
            adjacency_matrix (Tensor): The adjacency matrix.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        output_dict = self.forward(inputs, adjacency_matrix)
        losses = self.loss_by_feat(output_dict, batch_data_samples)
        return losses

    def predict(self, inputs: dict, 
                adjacency_matrix: Tensor,
                batch_input_metas: List[dict], 
                test_cfg: ConfigType
                ) -> Tensor:
        """Forward function for testing."""
        output_dict = self.forward(inputs, adjacency_matrix)
        if self.use_off_branch and self.co_shift != 0:
            output_dict['pred_offset'] /= self.co_shift
        return output_dict

    def forward_1branch(self, feat_dict: dict, adjacency_matrix: Tensor) -> dict:
        """Forward pass with semantic branch.
        
        Args:
            feat_dict (dict): Feature dict from backbone.
            adjacency_matrix (torch.Tensor): Adjacency matrix of shape [B, N, 3].

        Returns:
            dict: Outputs of semantic branch and offset branch.
                - seg_logit (torch.Tensor): Predicted segmentation map of shape 
                    [B, num_classes, N].
        """
        _, fa_points = self._extract_input(feat_dict)  # xyz_points,

        ### semantic branch
        sem_points = self.sem_FP_module(fa_points)

        ### graph-features modules
        for i in range(self.num_gf):
            sem_points = self.GF_modules[i](
                sem_points, adjacency_matrix=adjacency_matrix
            )  # 'guided_points=xyz_points' is not needed.
        
        ### features-propagation modules
        fp_points = self.join_FP_modules(sem_points)

        ### pre-segmentation conv
        fp_points = fp_points.transpose(1, 2).contiguous()  # [B, N, C] -> [B, C, N]
        fp_points = self.pre_seg_conv(fp_points)

        ### las cls conv
        seg_logit = self.cls_seg(fp_points)

        ### result dict
        output_dict = dict(seg_logit=seg_logit)

        return output_dict

    def forward_2branch(self, feat_dict: dict, adjacency_matrix: Tensor) -> dict:
        """Forward pass with semantic branch + offset branch.
        
        Args:
            feat_dict (dict): Feature dict from backbone.
            adjacency_matrix (torch.Tensor): Adjacency matrix of shape [B, N, 3].

        Returns:
            dict: Outputs of semantic branch and offset branch.
                - seg_logit (torch.Tensor): Predicted segmentation map of shape 
                    [B, num_classes, N].
                - pred_offset (torch.Tensor): Predicted offset of points of shape 
                    [B, N, 3].
                - origin_xyz (torch.Tensor): Original coords of points of shape 
                    [B, N, 3].
        """
        xyz_points, fa_points = self._extract_input(feat_dict)

        ### semantic branch
        sem_points = self.sem_FP_module(fa_points)

        ### offset branch
        off_points = self.off_FP_module(fa_points)
        off_points = self._pred_off(off_points)

        ### graph-features modules
        # origin code did NOT divide co_shift here
        shifted_points = xyz_points - off_points  # / self.co_shift
        for i in range(self.num_gf):
            sem_points = self.GF_modules[i](
                sem_points,
                guided_points=shifted_points,
                adjacency_matrix=adjacency_matrix
            )
        
        ### features-propagation modules
        fp_points = self.join_FP_modules(sem_points)

        ### pre-segmentation conv
        fp_points = fp_points.transpose(1, 2).contiguous()  # [B, N, C] -> [B, C, N]
        fp_points = self.pre_seg_conv(fp_points)

        ### las cls conv
        seg_logit = self.cls_seg(fp_points)

        ### result dict
        output_dict = dict(
            seg_logit=seg_logit,
            pred_offset=off_points,
            origin_xyz=xyz_points
        )

        return output_dict

    def loss_by_feat(self, output_dict: dict, batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Compute semantic segmentation loss.

        Args:
            output_dict (dict): Predicted outputs from decode head.
                - seg_logit (Tensor): Predicted per-point segmentation logits of
                    shape [B, num_classes, N].
                - pred_offset
                - origin_xyz
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        loss = dict()

        ### semantic branch loss
        seg_logit = output_dict['seg_logit']  # [B, cls, N]
        seg_label = self._stack_batch_gt(batch_data_samples)
        loss['loss_sem_seg'] = self.loss_semantic(
            seg_logit, seg_label, ignore_index=self.ignore_index
        )

        ### offset branch loss
        if self.use_off_branch:
            pred_offset = output_dict['pred_offset']  # [B, N, 3]
            origin_xyz = output_dict['origin_xyz']    # [B, N, 3]
            gt_offset = self._get_target_offset(origin_xyz, seg_label)

            ## offset norm loss
            valid = torch.where(seg_label != self.ignore_index)
            loss['loss_off_norm'] = self.loss_offset_norm(pred_offset[valid], gt_offset[valid])

            ## offset direction loss
            if self.use_direction_loss:
                gt_offset_dir = F.normalize(gt_offset, dim=-1)
                pred_offset_dir = F.normalize(pred_offset, dim=-1)
                cos_sim = nn.CosineSimilarity(dim=-1)(gt_offset_dir, pred_offset_dir)
                loss['loss_off_dir'] = (
                    torch.sum(-cos_sim[valid]) / (valid[0].shape[0] + 1e-8)
                )

        return loss

    def _pred_off(self, off_points):
        """Just a wrapper to forward self.off_conv (nn.Conv1d).

        Args:
            off_points (torch.Tensor): Features of shape [B, N, C].

        Returns:
            torch.Tensor: Predicted offsets of shape [B, N, 3].
        """
        off_points = off_points.transpose(1, 2).contiguous()
        off_points = self.off_conv(off_points)
        off_points = off_points.transpose(1, 2).contiguous()

        return off_points

    def _get_target_offset(self, origin_xyz, seg_label):
        """Calculate ground-truth offset of each points.

        Args:
            origin_xyz (torch.Tensor): Original coords of points of shape [B, N, 3].
            seg_label (torch.Tensor): Ground-truth segmentation label of shape [B, N].
        
        Returns:
            torch.Tensor: Ground-truth offsets of shape [B, N, 3].
        """
        batch_size = origin_xyz.size(0)
        target_offset = torch.zeros_like(origin_xyz)
        
        for b in range(batch_size):
            xyz = origin_xyz[b]
            mask = seg_label[b]
            cls_center = [torch.mean(xyz[torch.where(mask==c)], axis=-2) for c in range(self.num_classes)]
            cls_center = torch.stack(cls_center)
            target_offset[b] = xyz - cls_center[mask]

        # zero offset classes
        if self.zero_offset_classes is not None:
            if isinstance(self.zero_offset_classes, int):
                self.zero_offset_classes = range(self.zero_offset_classes)
            for zero_cls in self.zero_offset_classes:
                target_offset[torch.where(seg_label==zero_cls)] *= 0
        
        target_offset = self.co_shift * target_offset
        return target_offset

