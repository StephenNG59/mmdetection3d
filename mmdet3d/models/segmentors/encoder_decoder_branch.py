from typing import Dict, List, Union, Optional

import numpy as np
import torch
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet3d.structures import PointData
from ...structures.det3d_data_sample import (ForwardResults,
                                            OptSampleList, SampleList)
from ..utils import add_prefix
from .encoder_decoder import EncoderDecoder3D

@MODELS.register_module()
class EncoderDecoder3DBranch(EncoderDecoder3D):
    
    def __init__(self, **kwargs):
        super(EncoderDecoder3DBranch, self).__init__(**kwargs)

    def extract_feat(self, points: Tensor, adjacency_matrix: Tensor) -> dict:
        """Extract features from points."""
        x = self.backbone(points, adjacency_matrix)
        if self.with_neck:
            x = self.neck(x, adjacency_matrix)  # neck(x)
        return x
    
    def encode_decode(self, batch_inputs: Tensor,
                      adjacency_matrix: Tensor,
                      batch_input_metas: List[dict]
                      ) -> Tensor:
        """Encode points with backbone and decode into a semantic segmentation
        map of the same size as input.
        
        Args:
            batch_input (Tensor): Input point cloud sample
            adjacency_matrix (Tensor): Adjacency matrix of shape [B, N, 3].
            batch_input_metas (List[dict]): Meta information of a batch of
                samples.

        Returns:
            dict[str, Tensor]: Outputs of decode head. Note that key 
                `origin_xyz` is deleted (only used in loss calculation).
                
                - seg_logit (Tensor): Predicted segmentation map of 
                    shape [B, num_classes, N].
                - pred_offset (Tensor): Predicted offset of points of 
                    shape [B, N, 3].
        """
        x = self.extract_feat(batch_inputs, adjacency_matrix)
        output_dict = self.decode_head.predict(x, adjacency_matrix,
                                               batch_input_metas,
                                               self.test_cfg)
        output_dict.pop('origin_xyz', None)
        return output_dict
    
    def _decode_head_forward_train(
            self, batch_inputs_dict: dict,
            adjacency_matrix: Tensor,
            batch_data_samples: SampleList
            ) -> Dict[str, Tensor]:
        """Run forward function and calculate loss for decode head in training.

        Args:
            batch_input (Tensor): Input point cloud sample
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components for decode head.
        """
        losses = dict()
        loss_decode = self.decode_head.loss(batch_inputs_dict,
                                            adjacency_matrix,
                                            batch_data_samples,
                                            self.train_cfg)
        
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses
    
    def forward(self,
                inputs: Union[dict, List[dict]],
                # adjacency_matrix: Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
          tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
          processed to a list of :obj:`SegDataSample`.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (dict or List[dict]): Input sample dict which includes
                'points' and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor): Image tensor has shape (B, C, H, W).
            data_samples (List[:obj:`Det3DDataSample`], optional):
                The annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`Det3DDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        # TODO 测试 inputs is of type dict. keys: dict_keys(['points'])
        # if isinstance(inputs, dict):
        #     print('inputs is of type dict. keys:', inputs.keys())
        #     print('points shape:', inputs['points'].__len__())  # 4
        # else:
            # print('inputs is of type List[dict]. keys:', [input.keys() for input in inputs])
        # print('data_samples\'s length = ', data_samples.__len__())  # 4
        # print('data_samples[0].gt_pts_seg keys in EncoderDecoder3DBranch.forward():', data_samples[0].gt_pts_seg.keys())
        # data_samples[0].gt_pts_seg keys in EncoderDecoder3DBranch.forward(): ['pts_semantic_mask', 'adjacency_matrix']

        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    # forward_train() in old version
    def loss(self, batch_inputs_dict: dict,
            #  adjacency_matrix: Tensor,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            adjacency_matrix (Tensor): The adjacency matrix.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        # extract features using backbone
        points = torch.stack(batch_inputs_dict['points'])
        adjacency_matrix = torch.stack(batch_inputs_dict['adjacency_matrix'])
        
        x = self.extract_feat(points, adjacency_matrix)

        losses = dict()

        loss_decode = self._decode_head_forward_train(
            x, adjacency_matrix, batch_data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            raise NotImplementedError(
                'auxiliary head for branch-decoder not implemented.')
        
        if self.with_regularization_loss:
            loss_regularize = self._loss_regularization_forward_train()
            losses.update(loss_regularize)

        return losses

    def inference(self, points: Tensor, adjacency_matrix: Tensor,
                  batch_input_metas: List[dict],
                  rescale: bool, ) -> Tensor:
        """Inference (single scene...) with slide/whole style.
        
        Args:
            points (Tensor): Input points of shape [B, N, 3+C].
            batch_input_metas (List[dict]): Meta information of a batch of
                samples.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.

        Returns:
            dict[str, Tensor]: 
                The output segmentation map.
        """
        assert self.test_cfg.mode in ['whole'], \
            'Only support `whole` inference currently.'
        output_dict = self.encode_decode(points, adjacency_matrix,
                                         batch_input_metas,
                                        )
        return output_dict
    
    def predict(self,
                batch_inputs_dict: dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Simple test with single scene.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.
                Defaults to True.

        Returns:
            List[:obj:`Det3DDataSample`]: Segmentation results of the input
            points. Each Det3DDataSample usually contains:

            - ``pred_pts_seg`` (PointData): Prediction of 3D semantic
              segmentation.
            - ``pts_seg_logits`` (PointData): Predicted logits of 3D semantic
              segmentation before normalization.
        """
        seg_logits_list = []
        pred_offsets_list = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
        
        points = batch_inputs_dict['points']
        adjacency_matrix = batch_inputs_dict['adjacency_matrix']
        for point, adj_mat, input_meta in zip(points, adjacency_matrix, batch_input_metas):
            infered_result = self.inference(
                point.unsqueeze(0), adj_mat.unsqueeze(0), [input_meta], rescale
            )
            seg_logits_list.append(infered_result['seg_logit'][0])
            pred_offsets_list.append(infered_result['pred_offset'][0])

        return self.postprocess_result(seg_logits_list,
                                       pred_offsets_list,
                                       batch_data_samples)

    def postprocess_result(self, seg_logits_list: List[Tensor],
                           pred_offsets_list: List[Tensor],
                           batch_data_samples: SampleList):
        """Convert results list to `Det3DDataSample`.

        Args:
            seg_logits_list (List[Tensor]): List of segmentation results,
                seg_logits from model of each input point clouds sample.
            pred_offsets_list (List[Tensor]): List of predicted results,
                pred_offset from model of each input point clouds sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            List[:obj:`Det3DDataSample`]: Segmentation results of the input
            points. Each Det3DDataSample usually contains:

            - ``pred_pts_seg`` (PointData): Prediction of 3D semantic
              segmentation.
            - ``pts_seg_logits`` (PointData): Predicted logits of 3D semantic
              segmentation before normalization.
            - ``pts_pred_offset`` (PointData): Predicted offsets of points.
        """
        assert len(seg_logits_list) == len(pred_offsets_list)
        for i in range(len(seg_logits_list)):
            seg_logit = seg_logits_list[i]
            seg_pred = seg_logit.argmax(dim=0)
            pred_offset = pred_offsets_list[i]
            batch_data_samples[i].set_data({
                'pts_seg_logits':
                PointData(**{'pts_seg_logits': seg_logit}),
                'pred_pts_seg':
                PointData(**{'pts_semantic_mask': seg_pred}),
                'pts_pred_offset':
                PointData(**{'pts_pred_offset': pred_offset})
            })
        return batch_data_samples
        
    def _forward(self,
                 batch_inputs_dict: dict,
                 batch_data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.
        """
        points = torch.stack(batch_inputs_dict['points'])
        adjacency_matrix = torch.stack(batch_inputs_dict['adjacency_matrix'])
        
        x = self.extract_feat(points, adjacency_matrix)
        return self.decode_head.forward(x, adjacency_matrix)