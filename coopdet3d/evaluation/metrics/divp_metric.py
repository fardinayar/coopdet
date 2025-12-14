# Copyright (c) OpenMMLab. All rights reserved.
"""DIVP evaluation metric.

This module provides comprehensive evaluation for DIVP dataset.
All evaluation logic is self-contained - no delegation to dataset classes.

Supports two evaluation protocols:
1. NuScenes-style: Center distance matching + TP error metrics (mAP, mATE, mASE, mAOE, mAVE, NDS)
2. KITTI-style: IoU-based matching for BEV/3D mAP with difficulty levels

Coordinate System:
- All evaluation happens in RSU (infrastructure/registered) coordinate system
- Vehicle LiDAR points are transformed to infrastructure CS during data loading
- Ground truth annotations are in infrastructure CS

References:
- https://github.com/nutonomy/nuscenes-devkit
- http://www.cvlibs.net/datasets/kitti/eval_object.php
"""

import json
import os
import tempfile
import time
from os import path as osp
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.utils import mkdir_or_exist
from scipy.spatial.transform import Rotation

from mmdet3d.registry import METRICS

# Import evaluation functional utilities
from coopdet3d.evaluation.functional import (
    accumulate_center_distance,
    accumulate_iou,
    calc_ap,
    calc_tp_errors,
    filter_eval_boxes,
)


@METRICS.register_module()
@MMENGINE_METRICS.register_module()
class DIVPMetric(BaseMetric):
    """DIVP evaluation metric.

    This evaluator provides comprehensive evaluation for cooperative 3D detection
    on the DIVP dataset using both NuScenes-style and KITTI-style protocols.

    Args:
        data_root (str): Path to dataset root.
        ann_file (str): Path to annotation file (pickle).
        metric (str or List[str]): Metrics to evaluate. Default: 'bbox'.
        modality (dict): Sensor modality config. Default: dict(use_camera=False, use_lidar=True).
        prefix (str, optional): Prefix for metric names.
        format_only (bool): Only format results without evaluation. Default: False.
        jsonfile_prefix (str, optional): Prefix for output JSON files.
        result_names (List[str]): Result names. Default: ['pts_bbox'].
        dataset_type (str, optional): Dataset type ('TUMTrafV2XNuscDataset' or 'TUMTrafNuscDataset').
        collect_device (str): Device for collecting results ('cpu' or 'gpu'). Default: 'cpu'.
        backend_args (dict, optional): Backend arguments.
    """

    # =========================================================================
    # Class-level configuration
    # =========================================================================

    CLASSES = (
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

    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
    }

    # Detection range per class (meters)
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

    # NuScenes-style center distance evaluation config
    dist_fcn = "center_distance"
    dist_ths = [0.5, 1.0, 2.0, 4.0]  # Distance thresholds for AP calculation
    dist_th_tp = 2.0  # Threshold for TP error metrics

    # KITTI-style IoU evaluation config
    bev_iou_ths = [0.5, 0.7]  # BEV IoU thresholds for BEV mAP
    iou_3d_ths = [0.5, 0.7]   # 3D IoU thresholds for 3D mAP

    # General evaluation config
    min_recall = 0.1
    min_precision = 0.1
    max_boxes_per_sample = 500
    mean_ap_weight = 5  # Weight for mAP in NDS calculation

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 modality: dict = dict(use_camera=False, use_lidar=True),
                 prefix: Optional[str] = None,
                 format_only: bool = False,
                 jsonfile_prefix: Optional[str] = None,
                 result_names: List[str] = ['pts_bbox'],
                 dataset_type: Optional[str] = None,
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'DIVP metric'
        super(DIVPMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)

        self.ann_file = ann_file
        self.data_root = data_root
        self.modality = modality
        self.format_only = format_only

        if self.format_only:
            assert jsonfile_prefix is not None, \
                'jsonfile_prefix must be specified when format_only=True'

        self.jsonfile_prefix = jsonfile_prefix
        self.backend_args = backend_args
        self.result_names = result_names
        self.metrics = metric if isinstance(metric, list) else [metric]
        self.dataset_type = dataset_type

        # Load data_infos from annotation file for GT loading
        self.data_infos = None
        if osp.exists(osp.join(data_root, ann_file)):
            ann_data = load(osp.join(data_root, ann_file))
            # Support both 'infos' (old format) and 'data_list' (new format)
            if 'data_list' in ann_data:
                self.data_infos = ann_data['data_list']
            elif 'infos' in ann_data:
                self.data_infos = ann_data['infos']
            else:
                raise KeyError(f"Annotation file must contain either 'data_list' or 'infos' key. Found: {list(ann_data.keys())}")

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        Args:
            data_batch: Batch data from dataloader.
            data_samples: Batch outputs from model.
        """
        for data_sample in data_samples:
            result = dict()

            # Handle pred_instances_3d format (from mmengine runner)
            if 'pred_instances_3d' in data_sample:
                pred_3d = data_sample['pred_instances_3d']
                if 'bboxes_3d' in pred_3d:
                    result['boxes_3d'] = pred_3d['bboxes_3d'].to('cpu') if hasattr(
                        pred_3d['bboxes_3d'], 'to') else pred_3d['bboxes_3d']
                if 'scores' in pred_3d:
                    scores = pred_3d['scores']
                    result['scores_3d'] = scores.cpu() if hasattr(scores, 'cpu') else scores
                if 'labels' in pred_3d:
                    labels = pred_3d['labels']
                    result['labels_3d'] = labels.cpu() if hasattr(labels, 'cpu') else labels

            # Handle direct boxes_3d format
            elif 'boxes_3d' in data_sample:
                boxes = data_sample['boxes_3d']
                result['boxes_3d'] = boxes.to('cpu') if hasattr(boxes, 'to') else boxes
                if 'scores_3d' in data_sample:
                    scores = data_sample['scores_3d']
                    result['scores_3d'] = scores.cpu() if hasattr(scores, 'cpu') else scores
                if 'labels_3d' in data_sample:
                    labels = data_sample['labels_3d']
                    result['labels_3d'] = labels.cpu() if hasattr(labels, 'cpu') else labels

            # Store sample index
            if 'sample_idx' in data_sample:
                result['sample_idx'] = data_sample['sample_idx']
            elif 'img_path' in data_sample:
                result['sample_idx'] = data_sample.get('img_path', '')

            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute metrics from processed results.

        Args:
            results: Processed results from all batches.

        Returns:
            Dict of computed metrics.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # Format results to JSON
        result_files, tmp_dir = self._format_results(results)

        if self.format_only:
            logger.info(f'Results saved to {self.jsonfile_prefix}')
            if tmp_dir is not None:
                tmp_dir.cleanup()
            return {}

        # Run evaluation
        metric_dict = {}
        for metric in self.metrics:
            if isinstance(result_files, dict):
                for name in self.result_names:
                    logger.info(f"Evaluating {name}")
                    ret_dict = self._evaluate_single(
                        result_files[name],
                        logger=logger,
                        metric=metric,
                        result_name=name
                    )
                    metric_dict.update(ret_dict)
            elif isinstance(result_files, str):
                ret_dict = self._evaluate_single(
                    result_files,
                    logger=logger,
                    metric=metric,
                    result_name=self.result_names[0] if self.result_names else 'pts_bbox'
                )
                metric_dict.update(ret_dict)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return metric_dict

    # =========================================================================
    # Result formatting
    # =========================================================================

    def _format_results(self, results: List[dict]):
        """Format results to JSON files.

        Args:
            results: List of detection results.

        Returns:
            Tuple of (result_files, tmp_dir).
        """
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(self.data_infos), \
            f"Length mismatch: results={len(results)}, dataset={len(self.data_infos)}"

        if self.jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None
            jsonfile_prefix = self.jsonfile_prefix

        # Format to nuScenes-like JSON
        from coopdet3d.datasets.utils import output_to_box_dict, filter_box_in_lidar_cs

        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Converting detection format to JSON...")
        for index, det in enumerate(results):
            annos = []
            ts = str(self.data_infos[index]["timestamp"])
            boxes = output_to_box_dict(det)

            # Filter boxes in LiDAR coordinate system
            eval_configs = {
                "class_range": self.cls_range,
                "dist_fcn": self.dist_fcn,
                "dist_ths": self.dist_ths,
                "dist_th_tp": self.dist_th_tp,
            }
            boxes = filter_box_in_lidar_cs(boxes, mapped_class_names, eval_configs)

            for box in boxes:
                name = mapped_class_names[box["label"]]
                nusc_anno = dict(
                    timestamp=ts,
                    translation=box["center"].tolist(),
                    size=box["wlh"].tolist(),
                    rotation=float(box["orientation"]),  # Convert numpy to native Python
                    velocity=box["velocity"][:2].tolist(),
                    detection_name=name,
                    detection_score=float(box["score"]),  # Convert numpy to native Python
                )
                annos.append(nusc_anno)

            nusc_annos[ts] = annos

        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_nusc.json")
        print(f"Saving results to {res_path}")
        with open(res_path, 'w') as f:
            json.dump(nusc_submissions, f)

        return res_path, tmp_dir

    # =========================================================================
    # GT and prediction loading
    # =========================================================================

    def _load_gt(self, verbose: bool = False) -> Dict[str, List[dict]]:
        """Load ground truth boxes from OpenLabel annotations.

        Args:
            verbose: Whether to print progress.

        Returns:
            Dict mapping timestamp to list of GT boxes.
        """
        assert self.data_infos is not None and len(self.data_infos) > 0, \
            "No data_infos available. Check annotation file."

        all_annotations = {}

        for i, info in enumerate(self.data_infos):
            with open(info["lidar_anno_path"]) as f:
                lidar_annotation = json.load(f)

            # Get frame annotation
            lidar_anno_frame = None
            for frame_id in lidar_annotation['openlabel']['frames']:
                lidar_anno_frame = lidar_annotation['openlabel']['frames'][frame_id]
                break

            timestamp = str(lidar_anno_frame['frame_properties']['timestamp'])
            sample_boxes = []

            # Parse objects
            for obj_id in lidar_anno_frame['objects']:
                object_data = lidar_anno_frame['objects'][obj_id]['object_data']

                # Extract cuboid parameters
                loc = np.array(object_data['cuboid']['val'][:3], dtype=np.float32)
                dim = np.array(object_data['cuboid']['val'][7:], dtype=np.float32)
                rot = np.array(object_data['cuboid']['val'][3:7], dtype=np.float32)  # quaternion [x,y,z,w]

                # Convert quaternion to yaw (z-axis rotation)
                rot_obj = Rotation.from_quat(rot)
                rot_euler = rot_obj.as_euler('zyx', degrees=False)
                # IMPORTANT: Negate yaw to match the convention used in training data (pkl file)
                # The quaternion in JSON uses opposite rotation direction vs pkl GT boxes
                yaw = -float(rot_euler[0])  # First element is z-axis rotation, negated for convention match

                # Extract LiDAR points count
                num_lidar_pts = 0
                for attr in object_data['cuboid']['attributes']['num']:
                    if attr['name'] == 'num_points':
                        num_lidar_pts = attr['val']
                        break

                sample_boxes.append({
                    "timestamp": timestamp,
                    "translation": loc.tolist(),
                    "ego_dist": float(np.linalg.norm(loc[:2])),
                    "size": dim.tolist(),
                    "rotation": yaw,
                    "velocity": [0.0, 0.0],  # GT doesn't have velocity
                    "num_pts": int(num_lidar_pts),
                    "detection_name": object_data['type'],
                    "detection_score": -1.0,  # GT doesn't have score
                })

            all_annotations[timestamp] = sample_boxes

        if verbose:
            print(f"Loaded GT annotations for {len(all_annotations)} samples")

        return all_annotations

    def _load_predictions(
        self,
        result_path: str,
        max_boxes_per_sample: int,
        verbose: bool = False
    ) -> tuple:
        """Load predictions from JSON file.

        Args:
            result_path: Path to results JSON.
            max_boxes_per_sample: Max boxes per sample.
            verbose: Whether to print progress.

        Returns:
            Tuple of (predictions_dict, meta).
        """
        with open(result_path) as f:
            data = json.load(f)

        assert 'results' in data, 'No "results" field in result file'

        all_results = {}
        for idx, boxes in data['results'].items():
            box_list = []
            for box in boxes:
                box_list.append({
                    'timestamp': box['timestamp'],
                    'translation': box['translation'],
                    "ego_dist": float(np.linalg.norm(np.array(box['translation'][:2]))),
                    'size': box['size'],
                    'rotation': box['rotation'],
                    'velocity': box['velocity'],
                    'num_pts': -1 if 'num_pts' not in box else int(box['num_pts']),
                    'detection_name': box['detection_name'],
                    'detection_score': -1.0 if 'detection_score' not in box else float(box['detection_score'])
                })
            all_results[idx] = box_list

        # Check max boxes constraint
        for result in all_results:
            assert len(all_results[result]) <= max_boxes_per_sample, \
                f"Only <= {max_boxes_per_sample} boxes per sample allowed!"

        meta = data.get('meta', {})
        if verbose:
            print(f"Loaded predictions from {result_path}. Found {len(all_results)} samples.")

        return all_results, meta

    # =========================================================================
    # Main evaluation entry point
    # =========================================================================

    def _evaluate_single(
        self,
        result_path: str,
        logger: MMLogger,
        metric: str = 'bbox',
        result_name: str = 'pts_bbox'
    ) -> dict:
        """Evaluate single result file.

        Args:
            result_path: Path to result JSON file.
            logger: Logger instance.
            metric: Metric name.
            result_name: Result name.

        Returns:
            Dict of evaluation metrics.
        """
        output_dir = osp.dirname(result_path)

        # Run both evaluation protocols
        logger.info("Running NuScenes-style evaluation (center distance)")
        nusc_metrics = self._evaluate_nusc_style(result_path, output_dir, verbose=True)

        logger.info("Running KITTI-style evaluation (IoU-based BEV/3D mAP)")
        kitti_metrics = self._evaluate_kitti_style(result_path, output_dir, verbose=True)

        # Combine metrics
        all_metrics = {}

        # Add NuScenes metrics with prefix
        for key, value in nusc_metrics.items():
            all_metrics[f'{result_name}/{key}'] = value

        # Add KITTI metrics with prefix
        for key, value in kitti_metrics.items():
            all_metrics[f'{result_name}/{key}'] = value

        return all_metrics

    # =========================================================================
    # NuScenes-style evaluation (center distance)
    # =========================================================================

    def _evaluate_nusc_style(
        self,
        result_path: str,
        output_dir: str = None,
        verbose: bool = True
    ) -> dict:
        """Run NuScenes-style evaluation with center distance matching.

        Computes: mAP, mATE, mASE, mAOE, mAVE, NDS.

        Args:
            result_path: Path to predictions JSON.
            output_dir: Output directory for results.
            verbose: Whether to print progress.

        Returns:
            Dict of evaluation metrics.
        """
        assert osp.exists(result_path), f'Result file not found: {result_path}'

        if verbose:
            print('\n' + '='*70)
            print('TUMTraf NuScenes-style Evaluation (Center Distance Matching)')
            print('='*70)

        # Load data
        pred_boxes, meta = self._load_predictions(result_path, self.max_boxes_per_sample, verbose=verbose)
        gt_boxes = self._load_gt(verbose=verbose)

        assert set(pred_boxes.keys()) == set(gt_boxes.keys()), \
            "Sample mismatch between GT and predictions"

        # Filter boxes
        if verbose:
            print('Filtering predictions and GT by distance/points...')
        pred_boxes = filter_eval_boxes(pred_boxes, self.cls_range, verbose=verbose)
        gt_boxes = filter_eval_boxes(gt_boxes, self.cls_range, verbose=verbose)

        start_time = time.time()

        # Accumulate metrics for each class and distance threshold
        class_aps = {cls: [] for cls in self.CLASSES}
        class_tps = {cls: {metric: [] for metric in self.ErrNameMapping.keys()} for cls in self.CLASSES}

        for class_name in self.CLASSES:
            for dist_th in self.dist_ths:
                md = accumulate_center_distance(
                    gt_boxes, pred_boxes, class_name, dist_th, verbose=False
                )
                ap = calc_ap(md, self.min_recall, self.min_precision)
                class_aps[class_name].append(ap)

            # TP errors at dist_th_tp
            md_tp = accumulate_center_distance(
                gt_boxes, pred_boxes, class_name, self.dist_th_tp, verbose=False
            )
            for metric in self.ErrNameMapping.keys():
                tp_err = calc_tp_errors(md_tp, self.min_recall, metric)
                class_tps[class_name][metric].append(tp_err)

        eval_time = time.time() - start_time

        # Compute mean metrics
        mean_aps = {cls: float(np.mean(aps)) for cls, aps in class_aps.items()}
        mean_ap = float(np.mean(list(mean_aps.values())))

        tp_errors = {}
        for metric in self.ErrNameMapping.keys():
            cls_errors = [class_tps[cls][metric][0] for cls in self.CLASSES]
            tp_errors[metric] = float(np.mean(cls_errors))

        # Compute NDS (NuScenes Detection Score)
        tp_scores = [1 - tp_errors[metric] for metric in self.ErrNameMapping.keys()]
        nds = (self.mean_ap_weight * mean_ap + sum(tp_scores)) / (self.mean_ap_weight + len(tp_scores))

        # Prepare results
        metrics_summary = {
            'mAP': mean_ap,
            'NDS': float(nds),
            'eval_time': eval_time,
        }

        # Add TP errors with mapped names
        for metric, mapped_name in self.ErrNameMapping.items():
            metrics_summary[mapped_name] = tp_errors[metric]

        # Add per-class mAP
        for cls in self.CLASSES:
            metrics_summary[f'mAP_{cls}'] = mean_aps[cls]

        # Save to file
        if output_dir:
            metrics_summary['meta'] = meta
            metrics_summary['class_aps'] = class_aps
            metrics_summary['class_tps'] = class_tps
            with open(osp.join(output_dir, 'metrics_summary_nusc.json'), 'w') as f:
                # Convert numpy to native types
                def convert(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert(item) for item in obj]
                    else:
                        return obj
                json.dump(convert(metrics_summary), f, indent=2)

        # Print summary
        if verbose:
            print('\n' + '='*70)
            print('NuScenes-style Results')
            print('='*70)
            print(f'mAP:  {mean_ap:.4f}')
            print(f'NDS:  {nds:.4f}')
            for metric, mapped_name in self.ErrNameMapping.items():
                print(f'{mapped_name}: {tp_errors[metric]:.4f}')
            print(f'Eval time: {eval_time:.1f}s')
            print('\nPer-class mAP:')
            for cls in self.CLASSES:
                print(f'  {cls:<15} {mean_aps[cls]:.4f}')
            print('='*70)

        return metrics_summary

    # =========================================================================
    # KITTI-style evaluation (IoU-based)
    # =========================================================================

    def _evaluate_kitti_style(
        self,
        result_path: str,
        output_dir: str = None,
        verbose: bool = True
    ) -> dict:
        """Run KITTI-style evaluation with IoU matching.

        Computes: BEV mAP, 3D mAP (with difficulty levels).

        Args:
            result_path: Path to predictions JSON.
            output_dir: Output directory for results.
            verbose: Whether to print progress.

        Returns:
            Dict of evaluation metrics.
        """
        assert osp.exists(result_path), f'Result file not found: {result_path}'

        if verbose:
            print('\n' + '='*70)
            print('TUMTraf KITTI-style Evaluation (IoU-based)')
            print('='*70)

        # Load data
        pred_boxes, meta = self._load_predictions(result_path, self.max_boxes_per_sample, verbose=verbose)
        gt_boxes = self._load_gt(verbose=verbose)

        assert set(pred_boxes.keys()) == set(gt_boxes.keys()), \
            "Sample mismatch between GT and predictions"

        # Filter boxes
        if verbose:
            print('Filtering predictions and GT by distance/points...')
        pred_boxes = filter_eval_boxes(pred_boxes, self.cls_range, verbose=verbose)
        gt_boxes = filter_eval_boxes(gt_boxes, self.cls_range, verbose=verbose)

        start_time = time.time()

        # Compute BEV mAP
        if verbose:
            print('\nComputing BEV mAP...')
        bev_aps = {}
        for class_name in self.CLASSES:
            class_aps = []
            for iou_th in self.bev_iou_ths:
                md = accumulate_iou(
                    gt_boxes, pred_boxes, class_name, iou_th,
                    iou_type='bev', difficulty='all', verbose=False
                )
                ap = calc_ap(md, self.min_recall, self.min_precision)
                class_aps.append(ap)
            bev_aps[class_name] = class_aps

        # Compute 3D mAP
        if verbose:
            print('Computing 3D mAP...')
        iou_3d_aps = {}
        for class_name in self.CLASSES:
            class_aps = []
            for iou_th in self.iou_3d_ths:
                md = accumulate_iou(
                    gt_boxes, pred_boxes, class_name, iou_th,
                    iou_type='3d', difficulty='all', verbose=False
                )
                ap = calc_ap(md, self.min_recall, self.min_precision)
                class_aps.append(ap)
            iou_3d_aps[class_name] = class_aps

        eval_time = time.time() - start_time

        # Compute mean APs
        mean_bev_aps = {cls: float(np.mean(aps)) for cls, aps in bev_aps.items()}
        mean_3d_aps = {cls: float(np.mean(aps)) for cls, aps in iou_3d_aps.items()}
        overall_bev_map = float(np.mean(list(mean_bev_aps.values())))
        overall_3d_map = float(np.mean(list(mean_3d_aps.values())))

        # Prepare results
        metrics_summary = {
            'BEV_mAP': overall_bev_map,
            '3D_mAP': overall_3d_map,
            'eval_time': eval_time,
        }

        # Add per-class metrics
        for cls in self.CLASSES:
            metrics_summary[f'BEV_mAP_{cls}'] = mean_bev_aps[cls]
            metrics_summary[f'3D_mAP_{cls}'] = mean_3d_aps[cls]

        # Save to file
        if output_dir:
            metrics_summary['meta'] = meta
            metrics_summary['bev_aps'] = bev_aps
            metrics_summary['iou_3d_aps'] = iou_3d_aps
            metrics_summary['bev_iou_thresholds'] = self.bev_iou_ths
            metrics_summary['iou_3d_thresholds'] = self.iou_3d_ths
            with open(osp.join(output_dir, 'metrics_summary_kitti.json'), 'w') as f:
                def convert(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert(item) for item in obj]
                    else:
                        return obj
                json.dump(convert(metrics_summary), f, indent=2)

        # Print summary
        if verbose:
            print('\n' + '='*70)
            print('KITTI-style Results')
            print('='*70)
            print(f'BEV mAP: {overall_bev_map:.4f}')
            print(f'3D mAP:  {overall_3d_map:.4f}')
            print(f'Eval time: {eval_time:.1f}s')
            print('\nPer-class BEV mAP:')
            for cls in self.CLASSES:
                print(f'  {cls:<15} {mean_bev_aps[cls]:.4f}')
            print('\nPer-class 3D mAP:')
            for cls in self.CLASSES:
                print(f'  {cls:<15} {mean_3d_aps[cls]:.4f}')
            print('='*70)

        return metrics_summary
