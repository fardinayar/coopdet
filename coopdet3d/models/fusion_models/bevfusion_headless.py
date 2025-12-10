from typing import Any, Dict

import torch
from mmengine.model import BaseModule
from torch import nn
from torch.nn import functional as F

from coopdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_neck,
    build_vtransform,
    FUSIONMODELS,
)

# Import Voxelization and DynamicScatter from mmcv.ops
from mmcv.ops import Voxelization, DynamicScatter

__all__ = ["BEVFusionHeadless"]


@FUSIONMODELS.register_module()
class BEVFusionHeadless(BaseModule):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                print("USING VOXELIZE")
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                # DynamicScatter is currently not working
                print("USING DYNAMICSCATTER")
                if DynamicScatter is None:
                    raise ImportError("DynamicScatter is not available")
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)
        
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        vehicle2infrastructure,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)
        x = self.encoders["camera"]["vtransform"](
            self.training,
            x,
            points,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            vehicle2infrastructure,
            img_metas,
        )
        return x

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res.to(torch.float32))
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    def forward(
        self,
        img,
        points,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        vehicle2infrastructure,
        node,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs, batch_size = self.forward_single(
                img,
                points,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                vehicle2infrastructure,
                node,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs, batch_size

    def forward_single(
        self,
        img,
        points,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        vehicle2infrastructure,
        node,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        vehicle2infrastructure = vehicle2infrastructure.to(torch.float32)
        batch_size = len(points)
        
        # Transform every point cloud in the batch to the augmented infrastructure LiDAR perspective
        for b in range(batch_size):
            points[b] = points[b].to(torch.float32)

        features = []
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    vehicle2infrastructure,
                    metas,
                )
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]
            
        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]
            
        batch_size = x.shape[0]
        return x, batch_size

