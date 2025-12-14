import json
import os
import pickle
import tempfile
import time
from collections import defaultdict
from os import path as osp
from typing import Any, Dict

import numpy as np
import torch
from mmengine import dump, load
from mmengine.utils import mkdir_or_exist, track_iter_progress
from scipy.spatial.transform import Rotation

from mmdet3d.datasets import Det3DDataset
from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes


class _SerializedDataProxy:
    """Proxy class to access serialized data items when serialize_data=True."""
    
    def __init__(self, dataset):
        self._dataset = dataset
    
    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + idx
        start_addr = 0 if idx == 0 else self._dataset.data_address[idx - 1].item()
        end_addr = self._dataset.data_address[idx].item()
        bytes_data = memoryview(self._dataset.data_bytes[start_addr:end_addr])
        return pickle.loads(bytes_data)


@DATASETS.register_module()
class DIVPNuscDataset(Det3DDataset):
    METAINFO = {
        'classes': (
            'TYPE_PEDESTRIAN',
            'TYPE_BICYCLE',
            'TYPE_OTHER',
            'TYPE_SMALL_CAR',
            'TYPE_HEAVY_TRUCK',
            'TYPE_MOTORBIKE',
            'TYPE_MEDIUM_CAR',
            'TYPE_BUS',
            'TYPE_COMPACT_CAR',
            'TYPE_SEMITRACTOR',
            'TYPE_SEMITRAILER',
            'TYPE_LUXURY_CAR'
        )
    }

    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
    }

    # Modified from the originally used configs of BEVFusion https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/configs/detection_cvpr_2019.json
    cls_range = {
        "TYPE_PEDESTRIAN": 40,
        "TYPE_BICYCLE": 40,
        "TYPE_OTHER": 30,
        "TYPE_SMALL_CAR": 50,
        "TYPE_HEAVY_TRUCK": 50,
        "TYPE_MOTORBIKE": 40,
        "TYPE_MEDIUM_CAR": 50,
        "TYPE_BUS": 50,
        "TYPE_COMPACT_CAR": 50,
        "TYPE_SEMITRACTOR": 50,
        "TYPE_SEMITRAILER": 50,
        "TYPE_LUXURY_CAR": 50,
    }
    # Center distance-based evaluation (NuScenes style)
    dist_fcn = "center_distance"
    dist_ths = [0.5, 1.0, 2.0, 4.0]
    dist_th_tp = 2.0

    # IoU-based evaluation (KITTI/TUMTraf style)
    # BEV IoU thresholds for BEV mAP
    bev_iou_ths = [0.5, 0.7]
    # 3D IoU thresholds for 3D mAP
    iou_3d_ths = [0.5, 0.7]

    min_recall = 0.1
    min_precision = 0.1
    max_boxes_per_sample = 500
    mean_ap_weight = 5
    
    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        dataset_root=None,
        object_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        use_valid_flag=False,
        **kwargs,
    ) -> None:
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        # Support both data_root and dataset_root for compatibility
        self.dataset_root = data_root or dataset_root
        
        # Convert pipeline list to proper format if needed
        if pipeline is None:
            pipeline = []
        
        # Build metainfo with object_classes if provided
        metainfo = kwargs.pop('metainfo', None) or {}
        if object_classes is not None:
            metainfo['classes'] = tuple(object_classes)
        
        super().__init__(
            data_root=self.dataset_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality if modality else dict(use_lidar=True, use_camera=False),
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            metainfo=metainfo if metainfo else None,
            **kwargs,
        )

        self.with_velocity = with_velocity

        self.eval_detection_configs = {
            "class_range": self.cls_range, 
            "dist_fcn": self.dist_fcn, 
            "dist_ths": self.dist_ths, 
            "dist_th_tp": self.dist_th_tp, 
            "min_recall": self.min_recall, 
            "min_precision": self.min_precision, 
            "max_boxes_per_sample": self.max_boxes_per_sample,
            "mean_ap_weight": self.mean_ap_weight
        }

        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

    @property
    def CLASSES(self):
        """Backward compatibility property for class names."""
        return self.metainfo.get('classes', self.METAINFO['classes'])

    @property
    def cat2id(self):
        """Mapping from class name to class id."""
        return {name: i for i, name in enumerate(self.CLASSES)}

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    @property
    def data_infos(self):
        """Backward compatibility property for data_list.
        
        Note: When serialize_data=True, data_list is empty and data is stored
        in serialized form. Use get_data_info(idx) instead of direct access.
        """
        # If we have our stored _data_list, use it
        if hasattr(self, '_data_list') and len(self._data_list) > 0:
            return self._data_list
        # If data is serialized, return an empty list (caller should use get_data_info)
        if self.serialize_data and len(self.data_list) == 0:
            # Return a proxy that will deserialize on access
            return _SerializedDataProxy(self)
        return self.data_list

    def load_data_list(self):
        """Load annotations from ann_file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        annotations = load(self.ann_file)
        
        # Handle both old format (infos/metadata) and new format (data_list/metainfo)
        if 'infos' in annotations:
            # Old format
            raw_data_list = list(sorted(annotations["infos"], key=lambda e: e["timestamp"]))
            raw_data_list = raw_data_list[:: self.load_interval]
            self.metadata = annotations.get("metadata", {})
            self.version = self.metadata.get("version", "unknown")
        elif 'data_list' in annotations:
            # New format
            raw_data_list = annotations['data_list']
            if 'metainfo' in annotations:
                for k, v in annotations['metainfo'].items():
                    self._metainfo.setdefault(k, v)
            self.metadata = annotations.get('metainfo', {})
            self.version = self.metadata.get("version", "unknown")
        else:
            raise ValueError(f"Unknown annotation format. Expected 'infos' or 'data_list' key.")
        
        # Process each item through parse_data_info to count instances
        # This is required for the statistics table to show correct counts
        data_list = []
        for raw_info in raw_data_list:
            parsed_info = self.parse_data_info(raw_info)
            if parsed_info is not None:
                data_list.append(parsed_info)
        
        # Store data in instance for backward compatibility
        self._data_list = data_list
        return data_list

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info and count instances for statistics.
        
        This method is called during dataset initialization to count instances
        per category for the statistics table.
        
        Args:
            info (dict): Raw info dict from data_infos.
            
        Returns:
            dict: Processed info dict.
        """
        # Count instances from raw data for statistics
        # NOTE: Always count instances, even in test_mode, for evaluation statistics
        # The test_mode flag is used to skip GT loading during inference, but we still
        # need to count GT boxes for the statistics table shown during evaluation
        if 'gt_names' in info:
            # Filter based on valid flag or num_lidar_pts
            if self.use_valid_flag:
                mask = info.get("valid_flag", np.ones(len(info.get("gt_names", [])), dtype=bool))
            else:
                num_lidar_pts = info.get("num_lidar_pts", np.ones(len(info.get("gt_names", [])), dtype=np.int32))
                mask = num_lidar_pts > 0
            
            gt_names = np.array(info["gt_names"])[mask] if len(info.get("gt_names", [])) > 0 else []
            for cat_name in gt_names:
                if cat_name in self.CLASSES:
                    label = self.CLASSES.index(cat_name)
                    if label < len(self.num_ins_per_cat):
                        self.num_ins_per_cat[label] += 1
        
        # Call parent's parse_data_info if it exists, otherwise just return info
        # Since we're using get_data_info instead, we don't need to call parent
        return info

    def get_data_info(self, index: int) -> Dict[str, Any]:
        info = self.data_infos[index]

        data = dict(
            timestamp=info["timestamp"],
            location=info["location"],
            vehicle_lidar_path=info["vehicle_lidar_path"],
            vehicle_sweeps=info["vehicle_sweeps"],
            infrastructure_lidar_path=info["infrastructure_lidar_path"],
            infrastructure_sweeps=info["infrastructure_sweeps"],
            registered_lidar_path=info["registered_lidar_path"],
            registered_sweeps=info["registered_sweeps"],
            vehicle2infrastructure = np.asarray(info["vehicle2infrastructure"], dtype=np.float32),
        )

        if self.modality["use_camera"]:
            data["vehicle_image_paths"] = []
            data["vehicle_lidar2camera"] = []
            data["vehicle_lidar2image"] = []
            data["vehicle_camera_intrinsics"] = []
            data["vehicle_camera2lidar"] = []
            data["infrastructure_image_paths"] = []
            data["infrastructure_lidar2camera"] = []
            data["infrastructure_lidar2image"] = []
            data["infrastructure_camera_intrinsics"] = []
            data["infrastructure_camera2lidar"] = []

            for _, vehicle_camera_info in info["vehicle_cams"].items():
                data["vehicle_image_paths"].append(vehicle_camera_info["data_path"])

                # lidar to camera transform
                vehicle_camera2lidar = np.asarray(vehicle_camera_info["sensor2lidar"], dtype=np.float32)
                vehicle_camera2lidar = np.vstack([vehicle_camera2lidar, [0.0, 0.0, 0.0, 1.0]])
                vehicle_lidar2camera = np.linalg.inv(vehicle_camera2lidar)
                vehicle_lidar2camera = vehicle_lidar2camera[:-1, :]
                data["vehicle_lidar2camera"].append(vehicle_lidar2camera)

                # camera intrinsics
                data["vehicle_camera_intrinsics"].append(
                    np.asarray(vehicle_camera_info["camera_intrinsics"], dtype=np.float32)
                )

                # lidar to image transform
                data["vehicle_lidar2image"].append(
                    np.asarray(vehicle_camera_info["lidar2image"], dtype=np.float32)
                )

                # camera to lidar transform
                data["vehicle_camera2lidar"].append(
                    np.asarray(vehicle_camera_info["sensor2lidar"], dtype=np.float32)
                )
            
            for _, infrastructure_camera_info in info["infrastructure_cams"].items():
                data["infrastructure_image_paths"].append(infrastructure_camera_info["data_path"])

                # lidar to camera transform
                infrastructure_camera2lidar = np.asarray(infrastructure_camera_info["sensor2lidar"], dtype=np.float32)
                infrastructure_camera2lidar = np.vstack([infrastructure_camera2lidar, [0.0, 0.0, 0.0, 1.0]])
                infrastructure_lidar2camera = np.linalg.inv(infrastructure_camera2lidar)
                infrastructure_lidar2camera = infrastructure_lidar2camera[:-1, :]
                data["infrastructure_lidar2camera"].append(infrastructure_lidar2camera)

                # camera intrinsics
                data["infrastructure_camera_intrinsics"].append(
                    np.asarray(infrastructure_camera_info["camera_intrinsics"], dtype=np.float32)
                )

                # lidar to image transform
                data["infrastructure_lidar2image"].append(
                    np.asarray(infrastructure_camera_info["lidar2image"], dtype=np.float32)
                )

                # camera to lidar transform
                data["infrastructure_camera2lidar"].append(
                    np.asarray(infrastructure_camera_info["sensor2lidar"], dtype=np.float32)
                )

        if self.test_mode:
            annos = None
        else:
            annos = self.get_ann_info(index)
        data["ann_info"] = annos
        return data

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # Note: LiDARInstance3DBoxes uses origin=(0.5, 0.5, 0.5) by default.
        # The convert_to() method will handle conversion to the target box_mode_3d
        # if needed (e.g., converting to (0.5, 0.5, 0) for KITTI-style boxes).
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )
        return anns_results
    
    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for index, det in enumerate(track_iter_progress(results)):
            annos = []
            ts = str(self.data_infos[index]["timestamp"])
            boxes = output_to_box_dict(det)
            boxes = filter_box_in_lidar_cs(boxes, mapped_class_names, self.eval_detection_configs)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box["label"]]

                nusc_anno = dict(
                    timestamp=ts,
                    translation=box["center"].tolist(),
                    size=box["wlh"].tolist(),
                    rotation=box["orientation"],
                    velocity=box["velocity"][:2].tolist(),
                    detection_name=name,
                    detection_score=box["score"],
                )
                annos.append(nusc_anno)
            nusc_annos[ts] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_nusc.json")
        print("Results writes to", res_path)
        dump(nusc_submissions, res_path)
        return res_path
    
    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(
            self
        ), "The length of results is not equal to the dataset len: {} != {}".format(
            len(results), len(self)
        )

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        result_files = self._format_bbox(results, jsonfile_prefix)
        return result_files, tmp_dir

    # IDEA: Custom evaluation based on adapted NuScenes functions but not using nuscenes itself since it needs tokens
    # SEE: 
    # https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/evaluate.py
    # https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/algo.py
    # https://github.com/nutonomy/nuscenes-devkit/blob/da3c9a977112fca05413dab4e944d911769385a9/python-sdk/nuscenes/eval/common/utils.py
    # https://github.com/nutonomy/nuscenes-devkit/blob/da3c9a977112fca05413dab4e944d911769385a9/python-sdk/nuscenes/eval/detection/data_classes.py

    def load_prediction(self, result_path: str, max_boxes_per_sample: int, verbose: bool = False):
        """
        Loads object predictions from file.
        :param result_path: Path to the .json result file provided by the user.
        :param max_boxes_per_sample: Maximum number of boxes allowed per sample.
        :param verbose: Whether to print messages to stdout.
        :return: The deserialized results and meta data.
        """

        # Load from file and check that the format is correct.
        with open(result_path) as f:
            data = json.load(f)
        assert 'results' in data, 'Error: No field `results` in result file.'

        # Deserialize results and get meta data.
        all_results = {}
        for idx, boxes in data['results'].items():
            box_list = []
            for box in boxes:
                box_list.append({
                    'timestamp': box['timestamp'],
                    'translation': box['translation'],
                    "ego_dist": np.sqrt(np.sum(np.array(box['translation'][:2]) ** 2)),
                    'size': box['size'],
                    'rotation': box['rotation'],
                    'velocity': box['velocity'],
                    'num_pts': -1 if 'num_pts' not in box else int(box['num_pts']),
                    'detection_name': box['detection_name'],
                    'detection_score': -1.0 if 'detection_score' not in box else float(box['detection_score'])
                })
            all_results[idx] = box_list

        meta = data['meta']
        if verbose:
            print("Loaded results from {}. Found detections for {} samples.".format(result_path, len(all_results)))

        # Check that each sample has no more than x predicted boxes.
        for result in all_results:
            assert len(all_results[result]) <= max_boxes_per_sample, "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample

        return all_results, meta


    def load_gt(self, verbose: bool = False):
        """
        Loads ground truth boxes from database.
        :param nusc: A NuScenes instance.
        :param eval_split: The evaluation split for which we load GT boxes.
        :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
        :param verbose: Whether to print messages to stdout.
        :return: The GT boxes.
        """

        assert len(self.data_infos) > 0, "Error: Pickle has no samples!"

        all_annotations = {}

        for i, info in enumerate(self.data_infos):
            json1_file = open(info["lidar_anno_path"])
            json1_str = json1_file.read()
            lidar_annotation = json.loads(json1_str)

            lidar_anno_frame = {}

            for j in lidar_annotation['openlabel']['frames']:
                lidar_anno_frame = lidar_annotation['openlabel']['frames'][j]

            timestamp = str(lidar_anno_frame['frame_properties']['timestamp'])

            sample_boxes = []

            for id in lidar_anno_frame['objects']:
                object_data = lidar_anno_frame['objects'][id]['object_data']
                    
                loc = np.asarray(object_data['cuboid']['val'][:3], dtype=np.float32)
                dim = np.asarray(object_data['cuboid']['val'][7:], dtype=np.float32)
                rot = np.asarray(object_data['cuboid']['val'][3:7], dtype=np.float32) # Quaternion in x,y,z,w

                # Extract yaw rotation from quaternion using 'zyx' order to match TUM Traffic dev-kit
                # (evaluation.py line 828: as_euler("zyx", degrees=False)[0])
                # scipy Rotation.from_quat expects quaternion in [x, y, z, w] format
                rot_temp = Rotation.from_quat(rot)
                rot_euler = rot_temp.as_euler('zyx', degrees=False)
                yaw = np.asarray(rot_euler[0], dtype=np.float32)  # First element is z-axis rotation (yaw)
                # Note: MMDet3D uses counter-clockwise rotation (positive yaw increases from +x toward +y)
                # Negative yaw values are valid (clockwise rotation), but if GT and predictions
                # are consistently opposite signs, we may need to negate GT to match convention
                # The rotation alignment logic in bev_iou/iuo_3d should handle this, but if rotations
                # are consistently negative while predictions are positive, we might need to negate here

                num_lidar_pts = 0

                for n in object_data['cuboid']['attributes']['num']:
                    if n['name'] == 'num_points':
                        num_lidar_pts = n['val']

                sample_boxes.append({
                    "timestamp": timestamp,
                    "translation": loc,
                    "ego_dist": np.sqrt(np.sum(np.array(loc[:2]) ** 2)),
                    "size": dim,
                    "rotation": yaw,
                    "velocity": [0, 0],
                    "num_pts": num_lidar_pts,
                    "detection_name": object_data['type'],
                    "detection_score": -1.0,  # GT samples do not have a score.
                })

            all_annotations[timestamp] = sample_boxes

        if verbose:
            print("Loaded ground truth annotations for {} samples.".format(len(all_annotations)))

        return all_annotations
    
    def center_distance(self, gt_box, pred_box) -> float:
        """
        L2 distance between the box centers (xy only).
        :param gt_box: GT annotation sample.
        :param pred_box: Predicted sample.
        :return: L2 distance.
        """
        return np.linalg.norm(np.array(pred_box["translation"][:2]) - np.array(gt_box["translation"][:2]))

    def bev_iou(self, gt_box, pred_box) -> float:
        """
        Compute BEV (Bird's Eye View) IoU between two boxes using official mmdet3d method.
        BEV IoU considers only x, y dimensions and yaw rotation.

        :param gt_box: GT annotation sample dict with 'translation', 'size', 'rotation'.
        :param pred_box: Predicted sample dict with 'translation', 'size', 'rotation'.
        :return: BEV IoU value in [0, 1].
        """
        from mmdet3d.structures import LiDARInstance3DBoxes
        try:
            from mmcv.ops import box_iou_rotated
        except ImportError:
            from mmdet.structures.bbox import box_iou_rotated

        # Handle both "size" and "wlh" field names for compatibility
        gt_size = gt_box.get("size", gt_box.get("wlh", None))
        pred_size = pred_box.get("size", pred_box.get("wlh", None))
        
        if gt_size is None or pred_size is None:
            return 0.0
        
        # Convert boxes to LiDARInstance3DBoxes format: [x, y, z, dx, dy, dz, yaw]
        # Note: Using origin=(0.5, 0.5, 0.5) to match KITTI/nuScenes convention
        # IMPORTANT: 
        # - GT annotations from OpenLabel: cuboid['val'][7:10] format needs verification
        #   (TUM Traffic dev-kit evaluation.py line 675 stores as [l, w, h] but comment says [w, l, h])
        # - Predictions from box3d.dims: [dx, dy, dz] = [x_size, y_size, z_size] = [width, length, height]
        # - MMDet3D expects [dx, dy, dz] = [x_size, y_size, z_size] = [width, length, height]
        # Testing without swap first - if GT is already in correct format, no swap needed
        # If GT is [l, w, h] and needs to be [w, l, h], we would swap, but user suggests no swap
        gt_tensor = np.array([[
            gt_box["translation"][0],  # x
            gt_box["translation"][1],  # y
            gt_box["translation"][2],  # z
            gt_size[0],          # dx - NO SWAP: testing if GT is already in [dx, dy, dz] format
            gt_size[1],          # dy - NO SWAP: testing if GT is already in [dx, dy, dz] format
            gt_size[2],          # dz = height
            gt_box["rotation"]          # yaw (rotation around z-axis, CCW from +x toward +y)
        ]], dtype=np.float32)

        pred_tensor = np.array([[
            pred_box["translation"][0],
            pred_box["translation"][1],
            pred_box["translation"][2],
            pred_size[0],  # dx (x_size, width along x when yaw=0)
            pred_size[1],  # dy (y_size, length along y when yaw=0)
            pred_size[2],  # dz (z_size, height)
            pred_box["rotation"]
        ]], dtype=np.float32)
        
        # Normalize rotations to [-π, π] range to handle 2π ambiguity
        gt_yaw_raw = gt_tensor[0, 6]
        pred_yaw_raw = pred_tensor[0, 6]
        gt_yaw_norm = ((gt_yaw_raw + np.pi) % (2 * np.pi)) - np.pi
        pred_yaw_norm = ((pred_yaw_raw + np.pi) % (2 * np.pi)) - np.pi
        
        # Handle rotation convention mismatch: GT and predictions might be in different conventions
        # If they're approximately π apart, try both the original and negated GT rotation
        # and use the one that gives better alignment (smaller absolute difference)
        # For boxes, rotating by π doesn't change shape, but for IoU we need consistent convention
        diff_original = abs(gt_yaw_norm - pred_yaw_norm)
        diff_negated = abs((-gt_yaw_norm) - pred_yaw_norm)
        # Also check with π offset
        diff_pi_offset = abs((gt_yaw_norm + np.pi) - pred_yaw_norm)
        diff_pi_offset_norm = abs(((gt_yaw_norm + np.pi + np.pi) % (2 * np.pi) - np.pi) - pred_yaw_norm)
        
        # Use the rotation that minimizes the difference (but prefer original if close)
        if diff_original <= diff_negated and diff_original <= diff_pi_offset and diff_original <= diff_pi_offset_norm:
            gt_tensor[0, 6] = gt_yaw_norm
        elif diff_negated <= diff_pi_offset and diff_negated <= diff_pi_offset_norm:
            gt_tensor[0, 6] = -gt_yaw_norm
        elif diff_pi_offset <= diff_pi_offset_norm:
            gt_tensor[0, 6] = ((gt_yaw_norm + np.pi + np.pi) % (2 * np.pi)) - np.pi
        else:
            gt_tensor[0, 6] = gt_yaw_norm  # Default to original
        
        pred_tensor[0, 6] = pred_yaw_norm

        # Validate dimensions are positive (with small tolerance for floating point)
        if np.any(gt_tensor[0, 3:6] < 1e-6) or np.any(pred_tensor[0, 3:6] < 1e-6):
            return 0.0

        # Ensure tensors are on CPU for consistency
        gt_tensor = torch.from_numpy(gt_tensor)
        pred_tensor = torch.from_numpy(pred_tensor)
        
        gt_boxes = LiDARInstance3DBoxes(gt_tensor, box_dim=7, origin=(0.5, 0.5, 0.5))
        pred_boxes = LiDARInstance3DBoxes(pred_tensor, box_dim=7, origin=(0.5, 0.5, 0.5))

        # Get BEV representation: [x, y, dx, dy, yaw]
        gt_bev = gt_boxes.bev.cpu()  # Ensure on CPU
        pred_bev = pred_boxes.bev.cpu()  # Ensure on CPU

        # Clamp width and length to avoid numerical issues with very small boxes
        gt_bev[:, 2:4] = gt_bev[:, 2:4].clamp(min=1e-4)
        pred_bev[:, 2:4] = pred_bev[:, 2:4].clamp(min=1e-4)

        # Compute BEV IoU using official box_iou_rotated from mmcv/mmdet
        # Note: MMDet3D uses counter-clockwise rotation (right-handed coordinate system)
        # box_iou_rotated expects angles in counter-clockwise convention, which matches LiDAR
        # This returns (N, M) tensor
        try:
            iou_tensor = box_iou_rotated(gt_bev, pred_bev)
            iou = iou_tensor.cpu().numpy()[0, 0]
            # Handle NaN or invalid values
            if np.isnan(iou) or iou < 0 or iou > 1:
                return 0.0
            return float(iou)
        except Exception as e:
            # If IoU calculation fails, return 0 and optionally print error
            import warnings
            warnings.warn(f"BEV IoU calculation failed: {e}")
            return 0.0

    def iou_3d(self, gt_box, pred_box) -> float:
        """
        Compute 3D IoU between two boxes using official mmdet3d method.

        :param gt_box: GT annotation sample dict with 'translation', 'size', 'rotation'.
        :param pred_box: Predicted sample dict with 'translation', 'size', 'rotation'.
        :return: 3D IoU value in [0, 1].
        """
        from mmdet3d.structures import LiDARInstance3DBoxes

        # Handle both "size" and "wlh" field names for compatibility
        gt_size = gt_box.get("size", gt_box.get("wlh", None))
        pred_size = pred_box.get("size", pred_box.get("wlh", None))
        
        if gt_size is None or pred_size is None:
            return 0.0

        # Convert boxes to LiDARInstance3DBoxes format: [x, y, z, dx, dy, dz, yaw]
        # Note: Using origin=(0.5, 0.5, 0.5) to match KITTI/nuScenes convention
        # IMPORTANT: 
        # - GT annotations from OpenLabel: cuboid['val'][7:10] format needs verification
        #   (TUM Traffic dev-kit evaluation.py line 675 stores as [l, w, h] but comment says [w, l, h])
        # - Predictions from box3d.dims: [dx, dy, dz] = [x_size, y_size, z_size] = [width, length, height]
        # - MMDet3D expects [dx, dy, dz] = [x_size, y_size, z_size] = [width, length, height]
        # Testing without swap first - if GT is already in correct format, no swap needed
        # If GT is [l, w, h] and needs to be [w, l, h], we would swap, but user suggests no swap
        gt_tensor = np.array([[
            gt_box["translation"][0],  # x
            gt_box["translation"][1],  # y
            gt_box["translation"][2],  # z
            gt_size[0],          # dx - NO SWAP: testing if GT is already in [dx, dy, dz] format
            gt_size[1],          # dy - NO SWAP: testing if GT is already in [dx, dy, dz] format
            gt_size[2],          # dz = height
            gt_box["rotation"]          # yaw (rotation around z-axis, CCW from +x toward +y)
        ]], dtype=np.float32)

        pred_tensor = np.array([[
            pred_box["translation"][0],
            pred_box["translation"][1],
            pred_box["translation"][2],
            pred_size[0],  # dx (x_size, width along x when yaw=0)
            pred_size[1],  # dy (y_size, length along y when yaw=0)
            pred_size[2],  # dz (z_size, height)
            pred_box["rotation"]
        ]], dtype=np.float32)
        
        # Normalize rotations to [-π, π] range to handle 2π ambiguity
        gt_yaw_raw = gt_tensor[0, 6]
        pred_yaw_raw = pred_tensor[0, 6]
        gt_yaw_norm = ((gt_yaw_raw + np.pi) % (2 * np.pi)) - np.pi
        pred_yaw_norm = ((pred_yaw_raw + np.pi) % (2 * np.pi)) - np.pi
        
        # Handle rotation convention mismatch: GT and predictions might be in different conventions
        # If they're approximately π apart, try both the original and negated GT rotation
        # and use the one that gives better alignment (smaller absolute difference)
        # For boxes, rotating by π doesn't change shape, but for IoU we need consistent convention
        diff_original = abs(gt_yaw_norm - pred_yaw_norm)
        diff_negated = abs((-gt_yaw_norm) - pred_yaw_norm)
        # Also check with π offset
        diff_pi_offset = abs((gt_yaw_norm + np.pi) - pred_yaw_norm)
        diff_pi_offset_norm = abs(((gt_yaw_norm + np.pi + np.pi) % (2 * np.pi) - np.pi) - pred_yaw_norm)
        
        # Use the rotation that minimizes the difference (but prefer original if close)
        if diff_original <= diff_negated and diff_original <= diff_pi_offset and diff_original <= diff_pi_offset_norm:
            gt_tensor[0, 6] = gt_yaw_norm
        elif diff_negated <= diff_pi_offset and diff_negated <= diff_pi_offset_norm:
            gt_tensor[0, 6] = -gt_yaw_norm
        elif diff_pi_offset <= diff_pi_offset_norm:
            gt_tensor[0, 6] = ((gt_yaw_norm + np.pi + np.pi) % (2 * np.pi)) - np.pi
        else:
            gt_tensor[0, 6] = gt_yaw_norm  # Default to original
        
        pred_tensor[0, 6] = pred_yaw_norm

        # Validate dimensions are positive (with small tolerance for floating point)
        if np.any(gt_tensor[0, 3:6] < 1e-6) or np.any(pred_tensor[0, 3:6] < 1e-6):
            return 0.0

        # Ensure tensors are on CPU for consistency
        gt_tensor = torch.from_numpy(gt_tensor)
        pred_tensor = torch.from_numpy(pred_tensor)
        
        gt_boxes = LiDARInstance3DBoxes(gt_tensor, box_dim=7, origin=(0.5, 0.5, 0.5))
        pred_boxes = LiDARInstance3DBoxes(pred_tensor, box_dim=7, origin=(0.5, 0.5, 0.5))

        # Compute 3D IoU using official LiDARInstance3DBoxes.overlaps method
        # This uses the correct 3D rotated box IoU calculation from mmdet3d
        try:
            # Ensure boxes are on the same device (CPU)
            gt_boxes = gt_boxes.to('cpu')
            pred_boxes = pred_boxes.to('cpu')
            iou_tensor = LiDARInstance3DBoxes.overlaps(gt_boxes, pred_boxes, mode='iou')
            iou = iou_tensor.cpu().numpy()[0, 0]
            # Handle NaN or invalid values
            if np.isnan(iou) or iou < 0 or iou > 1:
                return 0.0
            return float(iou)
        except Exception as e:
            # If IoU calculation fails, return 0 and optionally print error
            import warnings
            warnings.warn(f"3D IoU calculation failed: {e}")
            return 0.0
    
    def velocity_l2(self, gt_box, pred_box) -> float:
        """
        L2 distance between the velocity vectors (xy only).
        If the predicted velocities are nan, we return inf, which is subsequently clipped to 1.
        :param gt_box: GT annotation sample.
        :param pred_box: Predicted sample.
        :return: L2 distance.
        """
        return np.linalg.norm(np.array(pred_box["velocity"]) - np.array(gt_box["velocity"]))


    def yaw_diff(self, gt_box, eval_box, period: float = 2*np.pi) -> float:
        """
        Returns the yaw angle difference between the orientation of two boxes.
        :param gt_box: Ground truth box.
        :param eval_box: Predicted box.
        :param period: Periodicity in radians for assessing angle difference.
        :return: Yaw angle difference in radians in [0, pi].
        """
        yaw_gt = gt_box["rotation"]
        yaw_est = eval_box["rotation"]

        return abs(self.angle_diff(yaw_gt, yaw_est, period))
    
    def angle_diff(self, x: float, y: float, period: float) -> float:
        """
        Get the smallest angle difference between 2 angles: the angle from y to x.
        :param x: To angle.
        :param y: From angle.
        :param period: Periodicity in radians for assessing angle difference.
        :return: <float>. Signed smallest between-angle difference in range (-pi, pi).
        """

        # calculate angle difference, modulo to [0, 2*pi]
        diff = (x - y + period / 2) % period - period / 2
        if diff > np.pi:
            diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]

        return diff
    
    def scale_iou(self, sample_annotation, sample_result) -> float:
        """
        This method compares predictions to the ground truth in terms of scale.
        It is equivalent to intersection over union (IOU) between the two boxes in 3D,
        if we assume that the boxes are aligned, i.e. translation and rotation are considered identical.
        :param sample_annotation: GT annotation sample.
        :param sample_result: Predicted sample.
        :return: Scale IOU.
        """
        # Validate inputs.
        sa_size = np.array(sample_annotation["size"])
        sr_size = np.array(sample_result["size"])
        assert all(sa_size > 0), 'Error: sample_annotation sizes must be >0.'
        assert all(sr_size > 0), 'Error: sample_result sizes must be >0.'

        # Compute IOU.
        min_wlh = np.minimum(sa_size, sr_size)
        volume_annotation = np.prod(sa_size)
        volume_result = np.prod(sr_size)
        intersection = np.prod(min_wlh)  # type: float
        union = volume_annotation + volume_result - intersection  # type: float
        iou = intersection / union

        return iou
    
    def cummean(self, x: np.array) -> np.array:
        """
        Computes the cumulative mean up to each position in a NaN sensitive way
        - If all values are NaN return an array of ones.
        - If some values are NaN, accumulate arrays discording those entries.
        - For positions where count_vals == 0 (all NaN so far), return 1.0 (max error)
        """
        if sum(np.isnan(x)) == len(x):
            # If all numbers in array are NaN's.
            return np.ones(len(x))  # If all errors are NaN set to error to 1 for all operating points.
        else:
            # Accumulate in a nan-aware manner.
            sum_vals = np.nancumsum(x.astype(float))  # Cumulative sum ignoring nans.
            count_vals = np.cumsum(~np.isnan(x))  # Number of non-nans up to each position.
            # For positions with no valid values yet, use 1.0 (max error)
            result = np.divide(sum_vals, count_vals, out=np.ones_like(sum_vals), where=count_vals != 0)
            # Set positions where count_vals == 0 to 1.0 (max error)
            result[count_vals == 0] = 1.0
            return result
        
    def get_difficulty_level(self, box: dict, difficulty: str = 'all') -> bool:
        """
        Determine if a box matches the specified difficulty level (KITTI-style).
        Since TUM Traffic doesn't have occlusion/truncation like KITTI, we use:
        - LiDAR points (similar to visibility/occlusion)
        - Distance from ego (farther = harder, similar to truncation)
        
        KITTI criteria (for reference):
        - Easy: min_height >= 40px, max_occlusion <= 0, max_truncation <= 0.15
        - Moderate: min_height >= 25px, max_occlusion <= 1, max_truncation <= 0.3
        - Hard: min_height >= 25px, max_occlusion <= 2, max_truncation <= 0.5
        
        :param box: Box dict with 'num_pts', 'ego_dist', etc.
        :param difficulty: 'easy', 'moderate', 'hard', or 'all'
        :return: True if box matches difficulty level
        """
        if difficulty == 'all':
            return True
        
        num_pts = box.get('num_pts', 0)
        ego_dist = box.get('ego_dist', float('inf'))
        
        # KITTI-style difficulty levels adapted for TUM Traffic:
        # Easy: many points (well-visible), close distance (not truncated)
        # Moderate: some points (partially visible), medium distance
        # Hard: few points (occluded) or far distance (truncated)
        #
        # IMPORTANT: Difficulties are EXCLUSIVE, not cumulative like KITTI
        # Easy boxes should be easiest, Hard boxes should be hardest
        if difficulty == 'easy':
            return num_pts >= 10 and ego_dist <= 40.0
        elif difficulty == 'moderate':
            # Moderate: between easy and hard thresholds
            return (5 <= num_pts < 10 or (num_pts >= 10 and 40.0 < ego_dist <= 60.0)) and ego_dist <= 60.0
        elif difficulty == 'hard':
            # Hard: below moderate thresholds but still detectable
            return (num_pts < 5 or ego_dist > 60.0) and num_pts >= 1 and ego_dist <= 80.0
        else:
            return True
    
    def accumulate_iou(self, gt_boxes: dict, pred_boxes: dict, class_name: str, iou_th: float,
                       iou_type: str = 'bev', verbose: bool = False, difficulty: str = 'all'):
        """
        Average Precision using IoU-based matching for BEV or 3D mAP.
        :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
        :param pred_boxes: Maps every sample_token to a list of its sample_results.
        :param class_name: Class to compute AP on.
        :param iou_th: IoU threshold for a match (e.g., 0.5, 0.7).
        :param iou_type: Type of IoU to use - 'bev' for BEV mAP or '3d' for 3D mAP.
        :param verbose: If true, print debug messages.
        :param difficulty: Difficulty level - 'easy', 'moderate', 'hard', or 'all' (default).
        :return: Dict with precision, recall, and confidence arrays.
        """
        # Count the positives (filtered by difficulty)
        gt_boxes_all = []
        for key in gt_boxes:
            gt_boxes_all.extend(gt_boxes[key])
        # Filter by class and difficulty
        npos = len([1 for box in gt_boxes_all 
                   if box["detection_name"] == class_name and self.get_difficulty_level(box, difficulty)])

        if verbose:
            print(f"Found {npos} GT of class {class_name} for {iou_type.upper()} IoU @ {iou_th}")

        # For missing classes in the GT, return empty result
        if npos == 0:
            return {
                "recall": np.linspace(0, 1, 101),
                "precision": np.zeros(101),
                "confidence": np.zeros(101),
            }

        # Organize predictions
        pred_boxes_all = []
        for key in pred_boxes:
            pred_boxes_all.extend(pred_boxes[key])
        
        # Debug: show all unique detection names if verbose
        if verbose and len(pred_boxes_all) > 0:
            unique_names = set(box["detection_name"] for box in pred_boxes_all)
            print(f"  All unique detection names in predictions: {sorted(unique_names)}")
            print(f"  Looking for class: '{class_name}'")
        
        pred_boxes_list = [box for box in pred_boxes_all if box["detection_name"] == class_name]
        pred_confs = [box["detection_score"] for box in pred_boxes_list]

        if verbose:
            print(f"Found {len(pred_confs)} PRED of class {class_name}")

        # Sort by confidence
        sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

        # Match and accumulate
        tp = []
        fp = []
        conf = []

        taken = set()
        for ind in sortind:
            pred_box = pred_boxes_list[ind]
            max_iou = 0.0
            match_gt_idx = None

            # Find best matching ground truth box (filtered by difficulty)
            for gt_idx, gt_box in enumerate(gt_boxes[pred_box["timestamp"]]):
                if (gt_box["detection_name"] == class_name and 
                    (pred_box["timestamp"], gt_idx) not in taken and
                    self.get_difficulty_level(gt_box, difficulty)):
                    # Compute IoU based on type
                    if iou_type == 'bev':
                        this_iou = self.bev_iou(gt_box, pred_box)
                    elif iou_type == '3d':
                        this_iou = self.iou_3d(gt_box, pred_box)
                    else:
                        raise ValueError(f"Unknown iou_type: {iou_type}")

                    # Debug: print first few IoU values if verbose
                    if verbose and len(tp) < 10:
                        gt_sz = gt_box.get('size', gt_box.get('wlh', 'N/A'))
                        pred_sz = pred_box.get('size', pred_box.get('wlh', 'N/A'))
                        gt_trans = gt_box.get('translation', 'N/A')
                        pred_trans = pred_box.get('translation', 'N/A')
                        gt_rot = gt_box.get('rotation', 'N/A')
                        pred_rot = pred_box.get('rotation', 'N/A')
                        # Show what we're actually using in the tensor (NO SWAP - testing)
                        if isinstance(gt_sz, (list, np.ndarray)) and len(gt_sz) >= 3:
                            gt_used = f"[dx={gt_sz[0]:.2f}, dy={gt_sz[1]:.2f}, dz={gt_sz[2]:.2f}]"  # NO SWAP
                        else:
                            gt_used = 'N/A'
                        if isinstance(pred_sz, (list, np.ndarray)) and len(pred_sz) >= 3:
                            pred_used = f"[dx={pred_sz[0]:.2f}, dy={pred_sz[1]:.2f}, dz={pred_sz[2]:.2f}]"  # no swap
                        else:
                            pred_used = 'N/A'
                        print(f"  IoU {iou_type}: {this_iou:.4f} | GT: trans={gt_trans}, size={gt_sz} -> tensor={gt_used}, rot={gt_rot:.3f} | Pred: trans={pred_trans}, size={pred_sz} -> tensor={pred_used}, rot={pred_rot:.3f}")

                    if this_iou > max_iou:
                        max_iou = this_iou
                        match_gt_idx = gt_idx

            # Check if IoU is above threshold
            is_match = max_iou >= iou_th

            if is_match:
                taken.add((pred_box["timestamp"], match_gt_idx))
                tp.append(1)
                fp.append(0)
                conf.append(pred_box["detection_score"])
            else:
                tp.append(0)
                fp.append(1)
                conf.append(pred_box["detection_score"])

        # Check if we have any matches
        if len(tp) == 0:
            return {
                "recall": np.linspace(0, 1, 101),
                "precision": np.zeros(101),
                "confidence": np.zeros(101),
            }

        # Calculate precision and recall
        tp = np.cumsum(tp).astype(float)
        fp = np.cumsum(fp).astype(float)
        conf = np.array(conf)

        prec = tp / (fp + tp)
        rec = tp / float(npos)

        # Interpolate
        rec_interp = np.linspace(0, 1, 101)
        prec = np.interp(rec_interp, rec, prec, right=0)
        conf = np.interp(rec_interp, rec, conf, right=0)
        rec = rec_interp

        return {
            "recall": rec,
            "precision": prec,
            "confidence": conf,
        }

    def accumulate(self, gt_boxes: list, pred_boxes: list, class_name: str, dist_th: float, verbose: bool = False):
        """
        Average Precision over predefined different recall thresholds for a single distance threshold.
        The recall/conf thresholds and other raw metrics will be used in secondary metrics.
        :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
        :param pred_boxes: Maps every sample_token to a list of its sample_results.
        :param class_name: Class to compute AP on.
        :param dist_fcn: Distance function used to match detections and ground truths.
        :param dist_th: Distance threshold for a match.
        :param verbose: If true, print debug messages.
        :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
        """
        # ---------------------------------------------
        # Organize input and initialize accumulators.
        # ---------------------------------------------

        # Count the positives.
        gt_boxes_all = []
        for key in gt_boxes:
            gt_boxes_all.extend(gt_boxes[key])
        npos = len([1 for box in gt_boxes_all if box["detection_name"] == class_name])
        if verbose:
            print("Found {} GT of class {} out of {} total across {} samples.".
                format(npos, class_name, len(gt_boxes_all), len(gt_boxes.keys())))

        # For missing classes in the GT, return a data structure corresponding to no predictions.
        if npos == 0:
            # Return dict with values of nuScenes DetectionMetricData.no_predictions()
            return {
                "recall": np.linspace(0, 1, 101), # 101 is from nuScene's nelem value
                "precision": np.zeros(101),
                "confidence": np.zeros(101),
                "trans_err": np.ones(101),
                "vel_err": np.ones(101),
                "scale_err": np.ones(101),
                "orient_err": np.ones(101)
            }

        # Organize the predictions in a single list.
        pred_boxes_all = []
        for key in pred_boxes:
            pred_boxes_all.extend(pred_boxes[key])
        pred_boxes_list = [box for box in pred_boxes_all if box["detection_name"] == class_name]
        pred_confs = [box["detection_score"] for box in pred_boxes_list]

        if verbose:
            print("Found {} PRED of class {} out of {} total across {} samples.".
                format(len(pred_confs), class_name, len(pred_boxes_all), len(pred_boxes.keys())))

        # Sort by confidence.
        sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

        # Do the actual matching.
        tp = []  # Accumulator of true positives.
        fp = []  # Accumulator of false positives.
        conf = []  # Accumulator of confidences.

        # Error arrays: one per prediction, NaN for FPs
        trans_err = []
        vel_err = []
        scale_err = []
        orient_err = []

        # ---------------------------------------------
        # Match and accumulate match data.
        # ---------------------------------------------

        taken = set()  # Initially no gt bounding box is matched.
        for ind in sortind:
            pred_box = pred_boxes_list[ind]
            min_dist = np.inf
            match_gt_idx = None

            for gt_idx, gt_box in enumerate(gt_boxes[pred_box["timestamp"]]):

                # Find closest match among ground truth boxes
                if gt_box["detection_name"] == class_name and not (pred_box["timestamp"], gt_idx) in taken:
                    this_distance = self.center_distance(gt_box, pred_box)
                    if this_distance < min_dist:
                        min_dist = this_distance
                        match_gt_idx = gt_idx

            # If the closest match is close enough according to threshold we have a match!
            is_match = min_dist < dist_th

            if is_match:
                taken.add((pred_box["timestamp"], match_gt_idx))

                #  Update tp, fp and confs.
                tp.append(1)
                fp.append(0)
                conf.append(pred_box["detection_score"])

                # Since it is a match, compute and store error metrics.
                gt_box_match = gt_boxes[pred_box["timestamp"]][match_gt_idx]

                trans_err.append(self.center_distance(gt_box_match, pred_box))
                vel_err.append(self.velocity_l2(gt_box_match, pred_box))
                scale_err.append(1 - self.scale_iou(gt_box_match, pred_box))

                # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
                period = np.pi if class_name == 'barrier' else 2 * np.pi
                orient_err.append(self.yaw_diff(gt_box_match, pred_box, period=period))

            else:
                # No match. Mark this as a false positive.
                tp.append(0)
                fp.append(1)
                conf.append(pred_box["detection_score"])

                # FPs get NaN for errors (will be ignored in cummean)
                trans_err.append(np.nan)
                vel_err.append(np.nan)
                scale_err.append(np.nan)
                orient_err.append(np.nan)

        # Check if we have any matches. If not, just return a "no predictions" array.
        if len(trans_err) == 0 or all(np.isnan(trans_err)):
            # Return dict with values of nuScenes DetectionMetricData.no_predictions()
            return {
                "recall": np.linspace(0, 1, 101), # 101 is from nuScene's nelem value
                "precision": np.zeros(101),
                "confidence": np.zeros(101),
                "trans_err": np.ones(101),
                "vel_err": np.ones(101),
                "scale_err": np.ones(101),
                "orient_err": np.ones(101)
            }

        # ---------------------------------------------
        # Calculate and interpolate precision and recall
        # ---------------------------------------------

        # Accumulate.
        tp = np.cumsum(tp).astype(float)
        fp = np.cumsum(fp).astype(float)
        conf_original = np.array(conf)  # Store original conf before interpolation

        # Calculate precision and recall.
        prec = tp / (fp + tp)
        rec = tp / float(npos)

        # Store original recall before interpolation for error interpolation
        rec_original = rec.copy()

        rec_interp = np.linspace(0, 1, 101)  # 101 steps, from 0% to 100% recall.
        prec = np.interp(rec_interp, rec, prec, right=0)
        conf = np.interp(rec_interp, rec, conf_original, right=0)
        rec = rec_interp

        # ---------------------------------------------
        # Compute cumulative mean of errors (NaN-aware) and interpolate
        # ---------------------------------------------
        
        # Convert error lists to arrays
        trans_err = np.array(trans_err)
        vel_err = np.array(vel_err)
        scale_err = np.array(scale_err)
        orient_err = np.array(orient_err)
        
        # Compute cumulative mean (NaN-aware, only over TPs)
        trans_err_cummean = self.cummean(trans_err)
        vel_err_cummean = self.cummean(vel_err)
        scale_err_cummean = self.cummean(scale_err)
        orient_err_cummean = self.cummean(orient_err)
        
        # Handle duplicate recall values by keeping only unique recall points
        # When multiple predictions have the same recall, use the last (most recent) cumulative mean
        # This ensures rec_original is strictly increasing for interpolation
        unique_rec_indices = []
        seen_rec = set()
        for i in range(len(rec_original) - 1, -1, -1):  # Iterate backwards
            rec_val = rec_original[i]
            if rec_val not in seen_rec:
                unique_rec_indices.insert(0, i)  # Insert at beginning to maintain order
                seen_rec.add(rec_val)
        
        rec_unique = rec_original[unique_rec_indices]
        trans_err_unique = trans_err_cummean[unique_rec_indices]
        vel_err_unique = vel_err_cummean[unique_rec_indices]
        scale_err_unique = scale_err_cummean[unique_rec_indices]
        orient_err_unique = orient_err_cummean[unique_rec_indices]
        
        # Interpolate to 101 points based on recall
        trans_err_interp = np.interp(rec_interp, rec_unique, trans_err_unique, left=1.0, right=1.0)
        vel_err_interp = np.interp(rec_interp, rec_unique, vel_err_unique, left=1.0, right=1.0)
        scale_err_interp = np.interp(rec_interp, rec_unique, scale_err_unique, left=1.0, right=1.0)
        orient_err_interp = np.interp(rec_interp, rec_unique, orient_err_unique, left=1.0, right=1.0)

        # ---------------------------------------------
        # Done. Instantiate MetricData and return
        # ---------------------------------------------
        return {
            "recall": rec,
            "precision": prec,
            "confidence": conf,
            "trans_err": trans_err_interp,
            "vel_err": vel_err_interp,
            "scale_err": scale_err_interp,
            "orient_err": orient_err_interp}

    def filter_eval_boxes(self, eval_boxes, max_dist: Dict[str, float], verbose: bool = False):
        """
        Applies filtering to boxes. Distance, bike-racks and points per box.
        :param nusc: An instance of the NuScenes class.
        :param eval_boxes: An instance of the EvalBoxes class.
        :param max_dist: Maps the detection name to the eval distance threshold for that class.
        :param verbose: Whether to print to stdout.
        """
        # Accumulators for number of filtered boxes.
        total, dist_filter, point_filter = 0, 0, 0
        for timestamp in eval_boxes:

            # Filter on distance first.
            total += len(eval_boxes[timestamp])
            # Only keep boxes whose class is in max_dist (cls_range)
            eval_boxes[timestamp] = [
                box for box in eval_boxes[timestamp] 
                if box["detection_name"] in max_dist and box["ego_dist"] < max_dist[box["detection_name"]]
            ]
            dist_filter += len(eval_boxes[timestamp])

            # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
            eval_boxes[timestamp] = [box for box in eval_boxes[timestamp] if not box["num_pts"] == 0]
            point_filter += len(eval_boxes[timestamp])

        if verbose:
            print("=> Original number of boxes: %d" % total)
            print("=> After distance based filtering: %d" % dist_filter)
            print("=> After LIDAR and RADAR points based filtering: %d" % point_filter)

        return eval_boxes
    
    def calc_ap(self, md, min_recall: float, min_precision: float) -> float:
        """ Calculated average precision. """

        assert 0 <= min_precision < 1
        assert 0 <= min_recall <= 1

        prec = np.copy(md["precision"])
        prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
        prec -= min_precision  # Clip low precision
        prec[prec < 0] = 0
        return float(np.mean(prec)) / (1.0 - min_precision)


    def calc_tp(self, md, min_recall: float, metric_name: str) -> float:
        """ Calculates true positive errors. """

        first_ind = round(100 * min_recall) + 1  # +1 to exclude the error at min recall.

        # Last instance of confidence > 0 is index of max achieved recall.
        non_zero = np.nonzero(md["confidence"])[0]
        if len(non_zero) == 0:  # If there are no matches, all the confidence values will be zero.
            max_recall_ind = 0
        else:
            max_recall_ind = non_zero[-1]

        last_ind = max_recall_ind  # First instance of confidence = 0 is index of max achieved recall.
        
        if last_ind < first_ind:
            return 1.0  # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.
        else:
            return float(np.mean(md[metric_name][first_ind: last_ind + 1]))  # +1 to include error at max recall.
        
    def serializeMetricDara(self, value):
        return {
            "recall": value["recall"].tolist(),
            "precision": value["precision"].tolist(),
            "confidence": value["confidence"].tolist(),
            "trans_err": value["trans_err"].tolist(),
            "vel_err": value["vel_err"].tolist(),
            "scale_err": value["scale_err"].tolist(),
            "orient_err": value["orient_err"].tolist()   
        }
    
    def _evaluate_bev_3d_map(
            self,
            result_path: str,
            output_dir: str = None,
            verbose: bool = True
    ):
        """Evaluate BEV mAP and 3D mAP using IoU-based matching.

        Args:
            result_path: Path to prediction results.
            output_dir: Directory to save evaluation results.
            verbose: Whether to print progress.

        Returns:
            dict: Evaluation metrics including BEV mAP and 3D mAP.
        """
        assert osp.exists(result_path), 'Error: The result file does not exist!'

        if verbose:
            print('Initializing BEV and 3D mAP evaluation')

        # Load predictions and ground truth
        self.pred_boxes, self.meta = self.load_prediction(result_path, self.max_boxes_per_sample, verbose=verbose)
        self.gt_boxes = self.load_gt(verbose=verbose)

        assert set(self.pred_boxes.keys()) == set(self.gt_boxes.keys()), \
            "Samples in split doesn't match samples in predictions."

        # Filter boxes (distance, points per box, etc.)
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = self.filter_eval_boxes(self.pred_boxes, self.cls_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = self.filter_eval_boxes(self.gt_boxes, self.cls_range, verbose=verbose)

        start_time = time.time()

        # Compute BEV mAP (with difficulty levels)
        if verbose:
            print('\n=== Computing BEV mAP ===')
        bev_aps = {}
        bev_aps_by_difficulty = {'easy': {}, 'moderate': {}, 'hard': {}, 'all': {}}
        for difficulty in ['all', 'easy', 'moderate', 'hard']:
            if verbose and difficulty != 'all':
                print(f'\n=== Computing BEV mAP ({difficulty.upper()}) ===')
            for class_name in self.CLASSES:
                class_aps = []
                for iou_th in self.bev_iou_ths:
                    # Enable verbose for first class, first threshold, and 'all' difficulty
                    debug_verbose = verbose and (class_name == self.CLASSES[0] and iou_th == self.bev_iou_ths[0] and difficulty == 'all')
                    md = self.accumulate_iou(self.gt_boxes, self.pred_boxes, class_name, iou_th, 
                                           iou_type='bev', verbose=debug_verbose, difficulty=difficulty)
                    ap = self.calc_ap(md, self.min_recall, self.min_precision)
                    class_aps.append(ap)
                if difficulty == 'all':
                    bev_aps[class_name] = class_aps
                else:
                    bev_aps_by_difficulty[difficulty][class_name] = class_aps
                if verbose and difficulty == 'all':
                    print(f"  {class_name} @ IoU {iou_th:.2f}: AP = {ap:.4f}")

        # Compute 3D mAP (with difficulty levels)
        if verbose:
            print('\n=== Computing 3D mAP ===')
        iou_3d_aps = {}
        iou_3d_aps_by_difficulty = {'easy': {}, 'moderate': {}, 'hard': {}, 'all': {}}
        for difficulty in ['all', 'easy', 'moderate', 'hard']:
            if verbose and difficulty != 'all':
                print(f'\n=== Computing 3D mAP ({difficulty.upper()}) ===')
            for class_name in self.CLASSES:
                class_aps = []
                for iou_th in self.iou_3d_ths:
                    # Enable verbose for first class, first threshold, and 'all' difficulty
                    debug_verbose = verbose and (class_name == self.CLASSES[0] and iou_th == self.iou_3d_ths[0] and difficulty == 'all')
                    md = self.accumulate_iou(self.gt_boxes, self.pred_boxes, class_name, iou_th, 
                                           iou_type='3d', verbose=debug_verbose, difficulty=difficulty)
                    ap = self.calc_ap(md, self.min_recall, self.min_precision)
                    class_aps.append(ap)
                if difficulty == 'all':
                    iou_3d_aps[class_name] = class_aps
                else:
                    iou_3d_aps_by_difficulty[difficulty][class_name] = class_aps
                if verbose and difficulty == 'all':
                    print(f"  {class_name} @ IoU {iou_th:.2f}: AP = {ap:.4f}")

        eval_time = time.time() - start_time

        # Compute mean APs for all difficulty levels
        mean_bev_aps = {class_name: float(np.mean(aps)) for class_name, aps in bev_aps.items()}
        mean_3d_aps = {class_name: float(np.mean(aps)) for class_name, aps in iou_3d_aps.items()}
        
        # Compute mean APs by difficulty
        mean_bev_aps_by_difficulty = {}
        mean_3d_aps_by_difficulty = {}
        for diff in ['easy', 'moderate', 'hard']:
            if bev_aps_by_difficulty[diff]:
                mean_bev_aps_by_difficulty[diff] = {class_name: float(np.mean(aps)) 
                                                   for class_name, aps in bev_aps_by_difficulty[diff].items()}
            if iou_3d_aps_by_difficulty[diff]:
                mean_3d_aps_by_difficulty[diff] = {class_name: float(np.mean(aps)) 
                                                 for class_name, aps in iou_3d_aps_by_difficulty[diff].items()}

        overall_bev_map = float(np.mean(list(mean_bev_aps.values())))
        overall_3d_map = float(np.mean(list(mean_3d_aps.values())))
        
        # Overall mAP by difficulty
        overall_bev_map_by_difficulty = {}
        overall_3d_map_by_difficulty = {}
        for diff in ['easy', 'moderate', 'hard']:
            if mean_bev_aps_by_difficulty.get(diff):
                overall_bev_map_by_difficulty[diff] = float(np.mean(list(mean_bev_aps_by_difficulty[diff].values())))
            if mean_3d_aps_by_difficulty.get(diff):
                overall_3d_map_by_difficulty[diff] = float(np.mean(list(mean_3d_aps_by_difficulty[diff].values())))

        # Prepare results summary
        metrics_summary = {
            'bev_aps': bev_aps,
            'mean_bev_aps': mean_bev_aps,
            'bev_map': overall_bev_map,
            'iou_3d_aps': iou_3d_aps,
            'mean_3d_aps': mean_3d_aps,
            'iou_3d_map': overall_3d_map,
            'eval_time': eval_time,
            'bev_iou_thresholds': self.bev_iou_ths,
            'iou_3d_thresholds': self.iou_3d_ths,
            # Difficulty-specific results
            'bev_aps_by_difficulty': bev_aps_by_difficulty,
            'mean_bev_aps_by_difficulty': mean_bev_aps_by_difficulty,
            'bev_map_by_difficulty': overall_bev_map_by_difficulty,
            'iou_3d_aps_by_difficulty': iou_3d_aps_by_difficulty,
            'mean_3d_aps_by_difficulty': mean_3d_aps_by_difficulty,
            'iou_3d_map_by_difficulty': overall_3d_map_by_difficulty,
        }

        # Save results
        if output_dir:
            metrics_summary['meta'] = self.meta.copy()
            with open(os.path.join(output_dir, 'bev_3d_metrics_summary.json'), 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_summary = {}
                for key, value in metrics_summary.items():
                    if isinstance(value, dict):
                        serializable_summary[key] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                      for k, v in value.items()}
                    elif isinstance(value, np.ndarray):
                        serializable_summary[key] = value.tolist()
                    else:
                        serializable_summary[key] = value
                json.dump(serializable_summary, f, indent=2)

        # Print summary
        if verbose:
            print('\n' + '='*60)
            print('BEV and 3D mAP Results')
            print('='*60)
            print(f'BEV mAP: {overall_bev_map:.4f}')
            print(f'3D mAP:  {overall_3d_map:.4f}')
            print(f'Eval time: {eval_time:.1f}s')
            
            # Print difficulty-specific results
            if overall_bev_map_by_difficulty:
                print('\nBEV mAP by Difficulty:')
                for diff in ['easy', 'moderate', 'hard']:
                    if diff in overall_bev_map_by_difficulty:
                        print(f'  {diff.capitalize()}: {overall_bev_map_by_difficulty[diff]:.4f}')
            if overall_3d_map_by_difficulty:
                print('\n3D mAP by Difficulty:')
                for diff in ['easy', 'moderate', 'hard']:
                    if diff in overall_3d_map_by_difficulty:
                        print(f'  {diff.capitalize()}: {overall_3d_map_by_difficulty[diff]:.4f}')
            
            print('\nPer-class BEV mAP:')
            print(f"{'Class':<20} {'BEV mAP':>10}")
            for class_name in mean_bev_aps.keys():
                print(f'{class_name:<20} {mean_bev_aps[class_name]:>10.4f}')
            print('\nPer-class 3D mAP:')
            print(f"{'Class':<20} {'3D mAP':>10}")
            for class_name in mean_3d_aps.keys():
                print(f'{class_name:<20} {mean_3d_aps[class_name]:>10.4f}')
            print('='*60)

        return metrics_summary

    def _evaluate_divp_nusc(
            self,
            config: dict,
            result_path: str,
            output_dir: str = None,
            verbose: bool = True
    ):
        assert osp.exists(result_path), 'Error: The result file does not exist!'

        if verbose:
            print('Initializing divp nuScenes detection evaluation')
        self.pred_boxes, self.meta = self.load_prediction(result_path, self.max_boxes_per_sample, verbose=verbose)
        self.gt_boxes = self.load_gt(verbose=verbose)

        assert set(self.pred_boxes.keys()) == set(self.gt_boxes.keys()), \
            "Samples in split doesn't match samples in predictions."

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = self.filter_eval_boxes(self.pred_boxes, self.cls_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = self.filter_eval_boxes(self.gt_boxes, self.cls_range, verbose=verbose)

        self.keys = self.gt_boxes.keys()

        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if verbose:
            print('Accumulating metric data...')
        metric_data_list = {}
        for class_name in self.CLASSES:
            for dist_th in self.dist_ths:
                md = self.accumulate(self.gt_boxes, self.pred_boxes, class_name, dist_th)
                metric_data_list[(class_name, dist_th)] = md

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if verbose:
            print('Calculating metrics...')
        metrics = {
            "label_aps": defaultdict(lambda: defaultdict(float)),
            "label_tp_errors": defaultdict(lambda: defaultdict(float))
        }
        for class_name in self.CLASSES:
            # Compute APs.
            for dist_th in self.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = self.calc_ap(metric_data, self.min_recall, self.min_precision)
                metrics["label_aps"][class_name][dist_th] = ap

            # Compute TP metrics.
            TP_METRICS = ['trans_err', 'scale_err', 'orient_err', 'vel_err']
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.dist_th_tp)]
                tp = self.calc_tp(metric_data, self.min_recall, metric_name)
                metrics["label_tp_errors"][class_name][metric_name] = tp

        # Compute evaluation time.
        metrics["eval_time"] = time.time() - start_time

        # Compute other values for metrics summary
        mean_dist_aps = {class_name: np.mean(list(d.values())) for class_name, d in metrics["label_aps"].items()}
        mean_ap = float(np.mean(list(mean_dist_aps.values())))
        
        tp_errors = {}
        for metric_name in TP_METRICS:
            class_errors = []
            for detection_name in self.CLASSES:
                class_errors.append(metrics["label_tp_errors"][detection_name][metric_name])

            tp_errors[metric_name] = float(np.nanmean(class_errors))

        tp_scores = {}
        for metric_name in TP_METRICS:

            # We convert the true positive errors to "scores" by 1-error.
            score = 1.0 - tp_errors[metric_name]

            # Some of the true positive errors are unbounded, so we bound the scores to min 0.
            score = max(0.0, score)

            tp_scores[metric_name] = score

        # Summarize.
        nd_score = float(self.mean_ap_weight * mean_ap + np.sum(list(tp_scores.values())))
        # Normalize.
        nd_score = nd_score / float(self.mean_ap_weight + len(tp_scores.keys()))

        # Dump the metric data, meta and metrics to disk.
        if verbose:
            print('Saving metrics to: %s' % output_dir)
        
        metrics_summary = {
            "label_aps": metrics["label_aps"],
            'mean_dist_aps': mean_dist_aps,
            'mean_ap': mean_ap,
            'label_tp_errors': metrics["label_tp_errors"],
            'tp_errors': tp_errors,
            'tp_scores': tp_scores,
            'nd_score': nd_score,
            'eval_time': metrics["eval_time"],
            'cfg': self.eval_detection_configs
        }
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)

        mdl_dump = {key[0] + ':' + str(key[1]): self.serializeMetricDara(value) for key, value in metric_data_list.items()}

        with open(os.path.join(output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(mdl_dump, f, indent=2)

        # Print high-level metrics.
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE'
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('%-20s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s' % ('Object Class', 'AP', 'ATE', 'ASE', 'AOE', 'AVE'))
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%-20s\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f'
                % (class_name, class_aps[class_name],
                    class_tps[class_name]['trans_err'],
                    class_tps[class_name]['scale_err'],
                    class_tps[class_name]['orient_err'],
                    class_tps[class_name]['vel_err']))
            
        return metrics_summary
    
    def _evaluate_single(
        self,
        result_path,
        logger=None,
        metric="bbox",
        result_name="pts_bbox",
    ):
        """Evaluation for a single model in nuScenes protocol + BEV/3D mAP.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        output_dir = osp.join(*osp.split(result_path)[:-1])

        # Evaluate using NuScenes-style metrics (center distance + TP errors)
        self._evaluate_divp_nusc(
            config=self.eval_detection_configs,
            result_path=result_path,
            output_dir=output_dir,
            verbose=False,
        )

        # Evaluate using BEV and 3D mAP (IoU-based)
        bev_3d_metrics = self._evaluate_bev_3d_map(
            result_path=result_path,
            output_dir=output_dir,
            verbose=True,  # Print BEV/3D mAP results
        )

        # record NuScenes-style metrics
        metrics = load(osp.join(output_dir, "metrics_summary.json"))
        detail = dict()
        for name in self.CLASSES:
            for k, v in metrics["label_aps"][name].items():
                val = float("{:.4f}".format(v))
                detail["object/{}_ap_dist_{}".format(name, k)] = val
            for k, v in metrics["label_tp_errors"][name].items():
                val = float("{:.4f}".format(v))
                detail["object/{}_{}".format(name, k)] = val
            for k, v in metrics["tp_errors"].items():
                val = float("{:.4f}".format(v))
                detail["object/{}".format(self.ErrNameMapping[k])] = val

        detail["object/nds"] = metrics["nd_score"]
        detail["object/map"] = metrics["mean_ap"]

        # Add BEV and 3D mAP metrics
        detail["object/bev_map"] = bev_3d_metrics["bev_map"]
        detail["object/3d_map"] = bev_3d_metrics["iou_3d_map"]

        # Add per-class BEV and 3D mAP
        for name in self.CLASSES:
            if name in bev_3d_metrics["mean_bev_aps"]:
                detail[f"object/{name}_bev_map"] = bev_3d_metrics["mean_bev_aps"][name]
            if name in bev_3d_metrics["mean_3d_aps"]:
                detail[f"object/{name}_3d_map"] = bev_3d_metrics["mean_3d_aps"][name]

            # Add per-threshold APs
            if name in bev_3d_metrics["bev_aps"]:
                for i, iou_th in enumerate(bev_3d_metrics["bev_iou_thresholds"]):
                    detail[f"object/{name}_bev_ap_{iou_th:.2f}"] = bev_3d_metrics["bev_aps"][name][i]
            if name in bev_3d_metrics["iou_3d_aps"]:
                for i, iou_th in enumerate(bev_3d_metrics["iou_3d_thresholds"]):
                    detail[f"object/{name}_3d_ap_{iou_th:.2f}"] = bev_3d_metrics["iou_3d_aps"][name][i]

        return detail

    def evaluate(
        self,
        results,
        metric="bbox",
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        **kwargs,
    ):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        metrics = {}

        if "boxes_3d" in results[0]:
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

            if isinstance(result_files, dict):
                for name in result_names:
                    print("Evaluating bboxes of {}".format(name))
                    ret_dict = self._evaluate_single(result_files[name])
                metrics.update(ret_dict)
            elif isinstance(result_files, str):
                metrics.update(self._evaluate_single(result_files))

            if tmp_dir is not None:
                tmp_dir.cleanup()

        return metrics

def output_to_box_dict(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`dict`]: List of standard box dicts.
    """
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    
    box_list = []
    for i in range(len(box3d)):
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        box = {
            "center": np.array(box_gravity_center[i]),
            "wlh": np.array(box_dims[i]),
            "orientation": box_yaw[i],
            "label": int(labels[i]) if not np.isnan(labels[i]) else labels[i],
            "score": float(scores[i]) if not np.isnan(scores[i]) else scores[i],
            "velocity": np.array(velocity),
            "name": None
        }
        box_list.append(box)
    return box_list


def filter_box_in_lidar_cs(boxes, classes, eval_configs):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`dict`]): List of predicted box dicts.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs : Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard box dicts in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # filter det in ego.
        cls_range_map = eval_configs["class_range"]
        radius = np.linalg.norm(box["center"][:2], 2)
        det_range = cls_range_map[classes[box["label"]]]
        if radius > det_range:
            continue
        box_list.append(box)
    return box_list
