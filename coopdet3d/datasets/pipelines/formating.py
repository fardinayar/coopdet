"""Formatting transforms for cooperative 3D detection."""
import numpy as np
import torch

from mmcv.transforms import to_tensor
from mmengine.structures import InstanceData

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import BaseInstance3DBoxes, Det3DDataSample
from mmdet3d.structures.points import BasePoints


@TRANSFORMS.register_module()
class DefaultFormatBundle3DCoop:
    """Default formatting bundle for cooperative 3D detection.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".

    Args:
        classes (list[str]): List of class names.
        with_gt (bool): Whether to include ground truth. Defaults to True.
        with_label (bool): Whether to include labels. Defaults to True.
    """

    def __init__(
        self,
        classes,
        with_gt: bool = True,
        with_label: bool = True,
    ) -> None:
        super().__init__()
        self.class_names = classes
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if "vehicle_points" in results:
            assert isinstance(results["vehicle_points"], BasePoints)
            results["vehicle_points"] = results["vehicle_points"].tensor

        if "infrastructure_points" in results:
            assert isinstance(results["infrastructure_points"], BasePoints)
            results["infrastructure_points"] = results["infrastructure_points"].tensor

        for key in ["voxels", "coors", "voxel_centers", "vehicle_num_points", "infrastructure_num_points"]:
            if key not in results:
                continue
            results[key] = to_tensor(results[key])

        if self.with_gt:
            # Clean GT bboxes in the final
            if "gt_bboxes_3d_mask" in results:
                gt_bboxes_3d_mask = results["gt_bboxes_3d_mask"]
                results["gt_bboxes_3d"] = results["gt_bboxes_3d"][gt_bboxes_3d_mask]
                if "gt_names_3d" in results:
                    results["gt_names_3d"] = results["gt_names_3d"][gt_bboxes_3d_mask]
                if "centers2d" in results:
                    results["centers2d"] = results["centers2d"][gt_bboxes_3d_mask]
                if "depths" in results:
                    results["depths"] = results["depths"][gt_bboxes_3d_mask]
            if "gt_bboxes_mask" in results:
                gt_bboxes_mask = results["gt_bboxes_mask"]
                if "gt_bboxes" in results:
                    results["gt_bboxes"] = results["gt_bboxes"][gt_bboxes_mask]
                results["gt_names"] = results["gt_names"][gt_bboxes_mask]
            if self.with_label:
                if "gt_names" in results and len(results["gt_names"]) == 0:
                    results["gt_labels"] = np.array([], dtype=np.int64)
                    results["attr_labels"] = np.array([], dtype=np.int64)
                elif "gt_names" in results and isinstance(results["gt_names"][0], list):
                    results["gt_labels"] = [
                        np.array(
                            [self.class_names.index(n) for n in res], dtype=np.int64
                        )
                        for res in results["gt_names"]
                    ]
                elif "gt_names" in results:
                    results["gt_labels"] = np.array(
                        [self.class_names.index(n) for n in results["gt_names"]],
                        dtype=np.int64,
                    )
                if "gt_names_3d" in results:
                    results["gt_labels_3d"] = np.array(
                        [self.class_names.index(n) for n in results["gt_names_3d"]],
                        dtype=np.int64,
                    )

        if "vehicle_img" in results:
            results["vehicle_img"] = torch.stack(results["vehicle_img"])

        if "infrastructure_img" in results:
            results["infrastructure_img"] = torch.stack(results["infrastructure_img"])

        for key in [
            "proposals",
            "gt_bboxes",
            "gt_bboxes_ignore",
            "gt_labels",
            "gt_labels_3d",
            "attr_labels",
            "centers2d",
            "depths",
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]
            else:
                results[key] = to_tensor(results[key])

        return results


@TRANSFORMS.register_module()
class Collect3DCoop:
    """Collect keys for cooperative 3D detection.

    Args:
        keys (list[str]): Keys to collect.
        meta_keys (tuple[str]): Meta keys to collect.
        meta_lis_keys (tuple[str]): Meta list keys to collect.
    """

    def __init__(
        self,
        keys,
        meta_keys=(
            "vehicle_camera_intrinsics",
            "vehicle_img_aug_matrix",
            "vehicle_lidar_aug_matrix",
            "infrastructure_camera_intrinsics",
            "infrastructure_img_aug_matrix",
            "infrastructure_lidar_aug_matrix",
            "vehicle2infrastructure",
        ),
        meta_lis_keys=(
            "timestamp",
            "vehicle_filename",
            "vehicle_lidar_path",
            "vehicle_ori_shape",
            "vehicle_img_shape",
            "vehicle_lidar2image",
            "infrastructure_filename",
            "infrastructure_lidar_path",
            "infrastructure_ori_shape",
            "infrastructure_img_shape",
            "infrastructure_lidar2image",
            "depth2img",
            "cam2img",
            "pad_shape",
            "scale_factor",
            "flip",
            "pcd_horizontal_flip",
            "pcd_vertical_flip",
            "box_mode_3d",
            "box_type_3d",
            "img_norm_cfg",
            "pcd_trans",
            "token",
            "pcd_scale_factor",
            "pcd_rotation",
            "transformation_3d_flow",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_lis_keys = meta_lis_keys

    def __call__(self, results):
        """Call function to collect keys in results and convert to new format.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - 'inputs' (dict): Model input data
                - 'data_samples' (Det3DDataSample): Annotation info
        """
        # Collect input data
        inputs = {}
        for key in self.keys:
            if key not in self.meta_keys and key in results:
                inputs[key] = results[key]

        # Collect meta keys as input data (for cooperative models)
        for key in self.meta_keys:
            if key in results:
                val = np.array(results[key])
                if isinstance(results[key], list) and key != "vehicle2infrastructure":
                    inputs[key] = to_tensor(val)
                elif isinstance(results[key], list) and key == "vehicle2infrastructure":
                    inputs[key] = to_tensor(val.astype(np.float32))
                else:
                    inputs[key] = to_tensor(val)

        # Create Det3DDataSample
        data_sample = Det3DDataSample()
        
        # Pack GT instances
        gt_instances_3d = InstanceData()
        if 'gt_bboxes_3d' in results:
            gt_instances_3d.bboxes_3d = results['gt_bboxes_3d']
        if 'gt_labels_3d' in results:
            gt_instances_3d.labels_3d = to_tensor(results['gt_labels_3d'])
        data_sample.gt_instances_3d = gt_instances_3d

        # Collect metadata
        metas = {}
        for key in self.meta_lis_keys:
            if key in results:
                metas[key] = results[key]

        # Also add meta_keys to metainfo for compatibility
        for key in self.meta_keys:
            if key in results:
                metas[key] = results[key]
        
        data_sample.set_metainfo(metas)

        # Return in new format
        packed_results = dict()
        packed_results['inputs'] = inputs
        packed_results['data_samples'] = data_sample
        return packed_results


__all__ = [
    'DefaultFormatBundle3DCoop',
    'Collect3DCoop',
]

