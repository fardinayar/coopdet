"""PointPillars encoder - matching original coopdet3d implementation."""
from typing import Any, Dict, Optional, Tuple

import torch
from mmcv.cnn import build_norm_layer
from torch import nn
from torch.nn import functional as F

from mmdet3d.registry import MODELS

__all__ = ['PillarFeatureNet', 'PointPillarsScatter', 'PointPillarsEncoder']


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.
    
    Args:
        actual_num: Actual number of points per voxel.
        max_num: Maximum number of points per voxel.
        axis: Axis to unsqueeze.
        
    Returns:
        Boolean mask tensor.
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(
        max_num_shape
    )
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator


class PFNLayer(nn.Module):
    """Pillar Feature Net Layer.
    
    The Pillar Feature Net could be composed of a series of these layers.
    This layer performs a similar role as VFELayer in SECOND.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        norm_cfg: Normalization config.
        last_layer: If last_layer, there is no concatenation of features.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_cfg: Optional[dict] = None,
        last_layer: bool = False,
    ):
        super().__init__()
        self.name = "PFNLayer"
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)
        self.norm_cfg = norm_cfg

        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.norm = build_norm_layer(self.norm_cfg, self.units)[1]

    def forward(self, inputs):
        x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        torch.backends.cudnn.enabled = True
        x = F.relu(x)

        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


@MODELS.register_module(force=True)
class PillarFeatureNet(nn.Module):
    """Pillar Feature Net - Original coopdet3d implementation.
    
    This implementation matches the original coopdet3d behavior:
    - Adds 5 feature decorations: f_cluster (3) + f_center (2) = 5
    - f_center only has x,y offset (not z)
    - Total features: in_channels + 5 (+ 1 if with_distance)
    
    Args:
        in_channels: Number of input features (e.g., x, y, z, intensity, time).
        feat_channels: Number of features in each PFNLayer.
        with_distance: Whether to include Euclidean distance to points.
        voxel_size: Size of voxels, only x and y are used.
        point_cloud_range: Point cloud range, only x and y min are used.
        norm_cfg: Normalization config.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        feat_channels: Tuple[int, ...] = (64,),
        with_distance: bool = False,
        voxel_size: Tuple[float, ...] = (0.2, 0.2, 4),
        point_cloud_range: Tuple[float, ...] = (0, -40, -3, 70.4, 40, 1),
        norm_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.name = "PillarFeatureNet"
        assert len(feat_channels) > 0

        self.in_channels = in_channels
        # Original coopdet3d adds 5 features: f_cluster(3) + f_center(2)
        in_channels += 5
        if with_distance:
            in_channels += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer
                )
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]

    def forward(self, features, num_voxels, coors):
        """Forward pass.
        
        Args:
            features: Voxel features [N, max_points, C].
            num_voxels: Number of points per voxel [N].
            coors: Voxel coordinates [N, 4] (batch, z, y, x) or [N, 3] (batch, x, y).
            
        Returns:
            Pillar features [N, feat_channels[-1]].
        """
        dtype = features.dtype

        # Find distance of x, y, z from cluster center
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(
            features
        ).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y from pillar center
        # After batch padding in voxelize(): coords = [batch, z, y, x]
        # So: coors[:, 3] = x_idx, coors[:, 2] = y_idx
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset
        )
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset
        )

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # Mask empty pillars
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)

        return features.squeeze()


@MODELS.register_module(force=True)
class PointPillarsScatter(nn.Module):
    """Point Pillar's Scatter - Original coopdet3d implementation.
    
    Converts learned features from dense tensor to sparse pseudo image.
    
    Args:
        in_channels: Number of input features.
        output_shape: Required output shape (nx, ny).
    """
    
    def __init__(
        self,
        in_channels: int = 64,
        output_shape: Tuple[int, int] = (512, 512),
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.output_shape = output_shape
        self.nx = output_shape[0]
        self.ny = output_shape[1]

    def extra_repr(self):
        return f"in_channels={self.in_channels}, output_shape={tuple(self.output_shape)}"

    def forward(self, voxel_features, coords, batch_size):
        """Forward pass.

        Args:
            voxel_features: Pillar features [N, C].
            coords: Voxel coordinates [N, 4] (batch, z, y, x) - modern mmdet3d format.
            batch_size: Batch size.

        Returns:
            BEV pseudo image [B, C, nx, ny] - original coopdet3d convention.
        """
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.in_channels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device,
            )

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt

            this_coords = coords[batch_mask, :]
            # Match original coopdet3d: coords = [batch, z, y, x]
            # Original uses column-major indexing: x_idx * ny + y_idx
            # coords[:, 3] = x_idx, coords[:, 2] = y_idx
            indices = this_coords[:, 3] * self.ny + this_coords[:, 2]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Scatter to canvas
            canvas[:, indices] = voxels

            batch_canvas.append(canvas)

        # Stack to 4-dim tensor - match original output shape [B, C, nx, ny]
        batch_canvas = torch.stack(batch_canvas, 0)
        batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.nx, self.ny)
        return batch_canvas


@MODELS.register_module(force=True)
class PointPillarsEncoder(nn.Module):
    """Encoder that combines PillarFeatureNet and PointPillarsScatter.
    
    Args:
        pts_voxel_encoder: Config for the voxel encoder (PillarFeatureNet).
        pts_middle_encoder: Config for the middle encoder (PointPillarsScatter).
    """
    
    def __init__(
        self,
        pts_voxel_encoder: Dict[str, Any],
        pts_middle_encoder: Dict[str, Any],
        **kwargs,
    ):
        super().__init__()
        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)

    def forward(self, feats, coords, batch_size, sizes):
        """Forward pass.
        
        Args:
            feats: Voxel features from voxelization.
            coords: Voxel coordinates.
            batch_size: Batch size.
            sizes: Number of points per voxel.
            
        Returns:
            BEV features from middle encoder.
        """
        x = self.pts_voxel_encoder(feats, sizes, coords)
        x = self.pts_middle_encoder(x, coords, batch_size)
        return x
