from typing import Any, Dict
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from coopdet3d.models.builder import (
    build_backbone,
    build_neck,
    build_coop_fuser,
    build_fusion_model_headless,
    build_head,
    COOPFUSIONMODELS,
)

from .base import Base3DCoopFusionModel

__all__ = ["BEVFusionCoop", "CooperativeTransFusionDetector"]


@COOPFUSIONMODELS.register_module()
class BEVFusionCoop(Base3DCoopFusionModel):
    def __init__(
        self,
        vehicle: Dict[str, Any],
        infrastructure: Dict[str, Any],
        coop_fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.nodes = nn.ModuleDict()
        if vehicle.get("fusion_model") is not None:
            self.nodes["vehicle"] = nn.ModuleDict(
                {
                    "fusion_model": build_fusion_model_headless(vehicle["fusion_model"])
                }
            )
        else:
            self.nodes["vehicle"] = None
        # Debug: Check infrastructure config
        if infrastructure is None:
            raise ValueError("infrastructure config is None. Make sure it's defined in the config file.")
        if not isinstance(infrastructure, dict):
            raise ValueError(f"infrastructure config must be a dict, got {type(infrastructure)}")
        if infrastructure.get("fusion_model") is not None:
            self.nodes["infrastructure"] = nn.ModuleDict(
                {
                    "fusion_model": build_fusion_model_headless(infrastructure["fusion_model"])
                }
            )
        else:
            self.nodes["infrastructure"] = None
            # Log warning about missing infrastructure fusion_model
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"infrastructure.fusion_model is None. Infrastructure keys: {list(infrastructure.keys()) if isinstance(infrastructure, dict) else 'N/A'}"
            )

        if coop_fuser is not None:
            self.coop_fuser = build_coop_fuser(coop_fuser)
        else:
            self.coop_fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )

        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0
        
        self.init_weights()

    def load_state_dict(self, state_dict, strict=True):
        """Override to filter out camera backbone keys before loading.
        
        Camera backbone weights are loaded separately via init_cfg from YOLOv8
        checkpoint, so we filter them out from the main checkpoint to avoid warnings.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Patterns for camera backbone keys that should be filtered
        camera_backbone_patterns = [
            'nodes.vehicle.fusion_model.encoders.camera.backbone',
            'nodes.infrastructure.fusion_model.encoders.camera.backbone',
            'nodes.vehicle.fusion_model.encoders.camera.neck',
            'nodes.infrastructure.fusion_model.encoders.camera.neck',
            'nodes.vehicle.fusion_model.encoders.camera.vtransform',
            'nodes.infrastructure.fusion_model.encoders.camera.vtransform',
        ]
        
        # Filter out camera-related keys from state_dict
        filtered_state_dict = {}
        camera_keys_removed = []
        
        for key, value in state_dict.items():
            is_camera_key = any(pattern in key for pattern in camera_backbone_patterns)
            if is_camera_key:
                camera_keys_removed.append(key)
            else:
                filtered_state_dict[key] = value
        
        if camera_keys_removed:
            logger.info(f"Filtered out {len(camera_keys_removed)} camera encoder keys from checkpoint "
                       f"(these are loaded from YOLOv8 checkpoint via init_cfg)")
            if len(camera_keys_removed) <= 10:
                logger.debug(f"Filtered keys (sample): {camera_keys_removed[:5]}...")
            else:
                logger.debug(f"Filtered keys (first 5 of {len(camera_keys_removed)}): {camera_keys_removed[:5]}...")
        
        # Load with filtered state_dict and strict=False to avoid warnings about missing camera keys
        # We use strict=False because we've already filtered out the camera keys
        return super().load_state_dict(filtered_state_dict, strict=False)

    def _extract_inputs(self, inputs):
        """Extract inputs from dict format to individual arguments.
        
        This method converts the inputs dict (from dataloader) to the
        individual arguments expected by forward_single().
        """
        return (
            inputs.get('vehicle_img'),
            inputs.get('vehicle_points'),
            inputs.get('vehicle_lidar2camera'),
            inputs.get('vehicle_lidar2image'),
            inputs.get('vehicle_camera_intrinsics'),
            inputs.get('vehicle_camera2lidar'),
            inputs.get('vehicle_img_aug_matrix'),
            inputs.get('vehicle_lidar_aug_matrix'),
            inputs.get('vehicle2infrastructure'),
            inputs.get('infrastructure_img'),
            inputs.get('infrastructure_points'),
            inputs.get('infrastructure_lidar2camera'),
            inputs.get('infrastructure_lidar2image'),
            inputs.get('infrastructure_camera_intrinsics'),
            inputs.get('infrastructure_camera2lidar'),
            inputs.get('infrastructure_img_aug_matrix'),
            inputs.get('infrastructure_lidar_aug_matrix'),
            inputs.get('metas', []),
        )

    def loss(self, inputs, data_samples=None, **kwargs):
        """Forward and return a dict of losses.
        
        This method follows mmdet3d's Base3DDetector interface pattern.
        
        Args:
            inputs (dict): Input data dict containing model inputs.
            data_samples (list, optional): Data samples. Defaults to None.
        
        Returns:
            dict: Dictionary of losses.
        """
        # Extract inputs
        args = self._extract_inputs(inputs)
        gt_bboxes_3d = inputs.get('gt_bboxes_3d')
        gt_labels_3d = inputs.get('gt_labels_3d')
        gt_masks_bev = inputs.get('gt_masks_bev')
        
        # Call forward_single in training mode
        self.train()
        outputs = self.forward_single(
            *args,
            gt_masks_bev=gt_masks_bev,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            **kwargs,
        )
        return outputs

    def predict(self, inputs, data_samples=None, **kwargs):
        """Forward and return predictions.
        
        This method follows mmdet3d's Base3DDetector interface pattern.
        
        Args:
            inputs (dict): Input data dict containing model inputs.
            data_samples (list, optional): Data samples. Defaults to None.
        
        Returns:
            list[dict]: List of prediction dictionaries, one per sample.
        """
        # Extract inputs
        args = self._extract_inputs(inputs)
        gt_masks_bev = inputs.get('gt_masks_bev')
        
        # Call forward_single in eval mode
        self.eval()
        outputs = self.forward_single(
            *args,
            gt_masks_bev=gt_masks_bev,
            gt_bboxes_3d=None,
            gt_labels_3d=None,
            **kwargs,
        )
        return outputs

    def _forward(self, inputs, data_samples=None, **kwargs):
        """Forward the whole network and return tensor or tuple of tensor.
        
        This method follows mmdet3d's Base3DDetector interface pattern.
        
        Args:
            inputs (dict): Input data dict containing model inputs.
            data_samples (list, optional): Data samples. Defaults to None.
        
        Returns:
            Tensor or tuple of Tensor: Raw model outputs without post-processing.
        """
        # Extract inputs
        args = self._extract_inputs(inputs)
        
        # Call forward_single without training/eval logic
        # This returns raw features
        features = []
        batch_size = 0
        
        vehicle_img, vehicle_points, vehicle_lidar2camera, vehicle_lidar2image, \
        vehicle_camera_intrinsics, vehicle_camera2lidar, vehicle_img_aug_matrix, \
        vehicle_lidar_aug_matrix, vehicle2infrastructure, infrastructure_img, \
        infrastructure_points, infrastructure_lidar2camera, infrastructure_lidar2image, \
        infrastructure_camera_intrinsics, infrastructure_camera2lidar, \
        infrastructure_img_aug_matrix, infrastructure_lidar_aug_matrix, metas = args
        
        # Process vehicle node
        if "vehicle" in self.nodes and self.nodes["vehicle"] is not None:
            feature, bs = self.nodes["vehicle"]["fusion_model"].forward(
        vehicle_img,
        vehicle_points,
        vehicle_lidar2camera,
        vehicle_lidar2image,
        vehicle_camera_intrinsics,
        vehicle_camera2lidar,
        vehicle_img_aug_matrix,
        vehicle_lidar_aug_matrix,
        vehicle2infrastructure,
                "vehicle",
                metas
            )
            features.append(feature)
            batch_size = bs
        
        # Process infrastructure node
        if "infrastructure" in self.nodes and self.nodes["infrastructure"] is not None:
            feature, bs = self.nodes["infrastructure"]["fusion_model"].forward(
        infrastructure_img,
        infrastructure_points,
        infrastructure_lidar2camera,
        infrastructure_lidar2image,
        infrastructure_camera_intrinsics,
        infrastructure_camera2lidar,
        infrastructure_img_aug_matrix,
        infrastructure_lidar_aug_matrix,
                vehicle2infrastructure,
                "infrastructure",
                metas
            )
            features.append(feature)
            batch_size = bs

        if self.coop_fuser is not None:
            if len(features) != 2:
                raise ValueError(f"coop_fuser expects 2 features, but got {len(features)}")
            x = self.coop_fuser(features)
        else:
            if len(features) != 1:
                raise ValueError(f"Expected 1 feature when coop_fuser is None, but got {len(features)}")
            x = features[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)
        
        return x

    def forward(
        self,
        vehicle_img=None,
        vehicle_points=None,
        vehicle_lidar2camera=None,
        vehicle_lidar2image=None,
        vehicle_camera_intrinsics=None,
        vehicle_camera2lidar=None,
        vehicle_img_aug_matrix=None,
        vehicle_lidar_aug_matrix=None,
        vehicle2infrastructure=None,
        infrastructure_img=None,
        infrastructure_points=None,
        infrastructure_lidar2camera=None,
        infrastructure_lidar2image=None,
        infrastructure_camera_intrinsics=None,
        infrastructure_camera2lidar=None,
        infrastructure_img_aug_matrix=None,
        infrastructure_lidar_aug_matrix=None,
        metas=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        mode='tensor',
        **kwargs,
    ):
        """The unified entry for a forward process.
        
        This method supports both the old interface (individual arguments) and
        the new interface (inputs dict with mode parameter).
        
        Args:
            mode (str): Return what kind of value. Defaults to 'tensor'.
                - 'loss': Forward and return a dict of losses.
                - 'predict': Forward and return predictions.
                - 'tensor': Forward and return tensor or tuple of tensor.
            ... (other arguments): Model inputs, either as individual arguments
                or as part of inputs dict.
        
        Returns:
            The return type depends on ``mode``.
        """
        # Support both old interface (individual args) and new interface (inputs dict)
        if isinstance(vehicle_img, dict) or (vehicle_img is None and 'inputs' in kwargs):
            # New interface: inputs dict
            if isinstance(vehicle_img, dict):
                inputs = vehicle_img
                mode = vehicle_points if isinstance(vehicle_points, str) else mode
            else:
                inputs = kwargs.pop('inputs', {})
                mode = kwargs.pop('mode', mode)
            
            # Extract data_samples from kwargs to avoid duplicate argument
            data_samples = kwargs.pop('data_samples', None)
            
            # Call parent's forward with mode
            return super().forward(inputs, data_samples=data_samples, mode=mode, **kwargs)
        
        # Old interface: individual arguments (for backward compatibility)
        # Handle batched inputs - stack images and transformation matrices, but keep point clouds as lists
        # Point clouds are handled by point encoders and should remain as lists
        if isinstance(vehicle_img, list):
            vehicle_img = torch.stack(vehicle_img)
        if isinstance(infrastructure_img, list):
            infrastructure_img = torch.stack(infrastructure_img)
        # Handle transformation matrices and intrinsics - should be tensors, not lists
        if isinstance(vehicle2infrastructure, list):
            vehicle2infrastructure = torch.stack(vehicle2infrastructure)
        if isinstance(vehicle_lidar2camera, list):
            vehicle_lidar2camera = torch.stack(vehicle_lidar2camera)
        if isinstance(vehicle_lidar2image, list):
            vehicle_lidar2image = torch.stack(vehicle_lidar2image)
        if isinstance(vehicle_camera_intrinsics, list):
            vehicle_camera_intrinsics = torch.stack(vehicle_camera_intrinsics)
        if isinstance(vehicle_camera2lidar, list):
            vehicle_camera2lidar = torch.stack(vehicle_camera2lidar)
        if isinstance(vehicle_img_aug_matrix, list):
            vehicle_img_aug_matrix = torch.stack(vehicle_img_aug_matrix)
        if isinstance(vehicle_lidar_aug_matrix, list):
            vehicle_lidar_aug_matrix = torch.stack(vehicle_lidar_aug_matrix)
        if isinstance(infrastructure_lidar2camera, list):
            infrastructure_lidar2camera = torch.stack(infrastructure_lidar2camera)
        if isinstance(infrastructure_lidar2image, list):
            infrastructure_lidar2image = torch.stack(infrastructure_lidar2image)
        if isinstance(infrastructure_camera_intrinsics, list):
            infrastructure_camera_intrinsics = torch.stack(infrastructure_camera_intrinsics)
        if isinstance(infrastructure_camera2lidar, list):
            infrastructure_camera2lidar = torch.stack(infrastructure_camera2lidar)
        if isinstance(infrastructure_img_aug_matrix, list):
            infrastructure_img_aug_matrix = torch.stack(infrastructure_img_aug_matrix)
        if isinstance(infrastructure_lidar_aug_matrix, list):
            infrastructure_lidar_aug_matrix = torch.stack(infrastructure_lidar_aug_matrix)
        # Point clouds should remain as lists - they're handled by point encoders
        # Don't stack them as they have different sizes per sample
        
        outputs = self.forward_single(
            vehicle_img,
            vehicle_points,
            vehicle_lidar2camera,
            vehicle_lidar2image,
            vehicle_camera_intrinsics,
            vehicle_camera2lidar,
            vehicle_img_aug_matrix,
            vehicle_lidar_aug_matrix,
            vehicle2infrastructure,
            infrastructure_img,
            infrastructure_points,
            infrastructure_lidar2camera,
            infrastructure_lidar2image,
            infrastructure_camera_intrinsics,
            infrastructure_camera2lidar,
            infrastructure_img_aug_matrix,
            infrastructure_lidar_aug_matrix,
            metas,
            gt_masks_bev,
            gt_bboxes_3d,
            gt_labels_3d,
            **kwargs,
        )
        return outputs

    def forward_single(
        self,
        vehicle_img,
        vehicle_points,
        vehicle_lidar2camera,
        vehicle_lidar2image,
        vehicle_camera_intrinsics,
        vehicle_camera2lidar,
        vehicle_img_aug_matrix,
        vehicle_lidar_aug_matrix,
        vehicle2infrastructure,
        infrastructure_img,
        infrastructure_points,
        infrastructure_lidar2camera,
        infrastructure_lidar2image,
        infrastructure_camera_intrinsics,
        infrastructure_camera2lidar,
        infrastructure_img_aug_matrix,
        infrastructure_lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        batch_size = 0
        
        # Process vehicle node
        if "vehicle" in self.nodes and self.nodes["vehicle"] is not None:
            feature, bs = self.nodes["vehicle"]["fusion_model"].forward(
                vehicle_img,
                vehicle_points,
                vehicle_lidar2camera,
                vehicle_lidar2image,
                vehicle_camera_intrinsics,
                vehicle_camera2lidar,
                vehicle_img_aug_matrix,
                vehicle_lidar_aug_matrix,
                vehicle2infrastructure,
                "vehicle",
                metas
            )
            features.append(feature)
            batch_size = bs
        
        # Process infrastructure node
        if "infrastructure" in self.nodes and self.nodes["infrastructure"] is not None:
            feature, bs = self.nodes["infrastructure"]["fusion_model"].forward(
                infrastructure_img,
                infrastructure_points,
                infrastructure_lidar2camera,
                infrastructure_lidar2image,
                infrastructure_camera_intrinsics,
                infrastructure_camera2lidar,
                infrastructure_img_aug_matrix,
                infrastructure_lidar_aug_matrix,
                vehicle2infrastructure,
                "infrastructure",
                metas
            )
            features.append(feature)
            batch_size = bs

        if self.coop_fuser is not None:
            if len(features) != 2:
                vehicle_status = "configured" if ("vehicle" in self.nodes and self.nodes["vehicle"] is not None) else "NOT configured"
                infra_status = "configured" if ("infrastructure" in self.nodes and self.nodes["infrastructure"] is not None) else "NOT configured"
                raise ValueError(
                    f"coop_fuser expects 2 features, but got {len(features)}. "
                    f"Vehicle node: {vehicle_status}, Infrastructure node: {infra_status}. "
                    f"Make sure both vehicle and infrastructure nodes are configured."
                )
            x = self.coop_fuser(features)
        else:
            if len(features) != 1:
                raise ValueError(
                    f"Expected 1 feature when coop_fuser is None, but got {len(features)}. "
                    f"Features: {[f.shape for f in features]}"
                )
            x = features[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        # DEBUG: Verify output format (first batch, first evaluation only)
                        if not hasattr(self, '_debug_model_output_logged') and k == 0:
                            import logging
                            import builtins
                            debug_logger = logging.getLogger(__name__)
                            debug_logger.warning("=" * 60)
                            debug_logger.warning("Model Output Format Check (first eval batch):")
                            debug_logger.warning(f"  Batch {k}:")
                            # Use builtins.type() to avoid shadowing by loop variable 'type'
                            debug_logger.warning(f"    boxes type: {builtins.type(boxes)}")
                            if hasattr(boxes, 'tensor'):
                                debug_logger.warning(f"    boxes.tensor shape: {boxes.tensor.shape}")
                                debug_logger.warning(f"    boxes.tensor dtype: {boxes.tensor.dtype}")
                                if len(boxes) > 0:
                                    debug_logger.warning(f"    boxes.tensor sample (first box): {boxes.tensor[0].tolist()}")
                            debug_logger.warning(f"    scores shape: {scores.shape}, dtype: {scores.dtype}")
                            debug_logger.warning(f"    labels shape: {labels.shape}, dtype: {labels.dtype}")
                            debug_logger.warning(f"    Number of predictions: {len(boxes) if hasattr(boxes, '__len__') else 'N/A'}")
                            debug_logger.warning("=" * 60)
                            self._debug_model_output_logged = True
                        
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs


# Alias for backward compatibility and clearer naming
@COOPFUSIONMODELS.register_module(name='CooperativeTransFusionDetector')
class CooperativeTransFusionDetector(BEVFusionCoop):
    """Alias for BEVFusionCoop with a more descriptive name.
    
    This class is identical to BEVFusionCoop but provides a clearer name
    that indicates it's a cooperative TransFusion-based detector.
    """
    pass

