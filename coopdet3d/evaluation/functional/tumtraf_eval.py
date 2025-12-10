# Copyright (c) OpenMMLab. All rights reserved.
"""TUMTraf V2X evaluation functional utilities.

This module contains pure evaluation functions for TUMTraf V2X cooperative perception dataset.
All evaluation happens in RSU (infrastructure) coordinate system.

The evaluation supports two protocols:
1. NuScenes-style: Center distance-based matching with TP error metrics (mATE, mASE, mAOE, mAVE, NDS)
2. KITTI-style: IoU-based matching for BEV mAP and 3D mAP with difficulty levels

References:
- NuScenes devkit: https://github.com/nutonomy/nuscenes-devkit
- KITTI evaluation: http://www.cvlibs.net/datasets/kitti/eval_object.php
- TUMTraf-V2X dataset: https://tum-traffic-dataset.github.io/tumtraf-v2x/
"""

import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
from mmdet3d.structures import LiDARInstance3DBoxes

try:
    from mmcv.ops import box_iou_rotated
except ImportError:
    from mmdet.structures.bbox import box_iou_rotated


# =============================================================================
# Distance and IoU computation functions
# =============================================================================

def center_distance(gt_box: dict, pred_box: dict) -> float:
    """Compute L2 distance between box centers (XY plane only).

    This is used for NuScenes-style evaluation matching.

    Args:
        gt_box: Ground truth box dict with 'translation' key.
        pred_box: Predicted box dict with 'translation' key.

    Returns:
        L2 distance in meters.
    """
    return np.linalg.norm(
        np.array(pred_box["translation"][:2]) - np.array(gt_box["translation"][:2])
    )


def bev_iou(gt_box: dict, pred_box: dict) -> float:
    """Compute Bird's Eye View IoU between two 3D boxes.

    Uses mmdet3d's official box_iou_rotated for accurate rotated rectangle IoU.
    BEV IoU considers only x, y dimensions and yaw rotation, ignoring height.

    Args:
        gt_box: GT box dict with 'translation', 'size'/'wlh', 'rotation' keys.
        pred_box: Predicted box dict with same structure.

    Returns:
        BEV IoU value in [0, 1].
    """
    # Handle both "size" and "wlh" field names
    gt_size = gt_box.get("size", gt_box.get("wlh"))
    pred_size = pred_box.get("size", pred_box.get("wlh"))

    if gt_size is None or pred_size is None:
        return 0.0

    # Build tensors: [x, y, z, dx, dy, dz, yaw]
    # Using origin=(0.5, 0.5, 0.5) to match KITTI/NuScenes convention
    gt_tensor = np.array([[
        gt_box["translation"][0],  # x
        gt_box["translation"][1],  # y
        gt_box["translation"][2],  # z
        gt_size[0],                # dx (width)
        gt_size[1],                # dy (length)
        gt_size[2],                # dz (height)
        gt_box["rotation"]         # yaw
    ]], dtype=np.float32)

    pred_tensor = np.array([[
        pred_box["translation"][0],
        pred_box["translation"][1],
        pred_box["translation"][2],
        pred_size[0],
        pred_size[1],
        pred_size[2],
        pred_box["rotation"]
    ]], dtype=np.float32)

    # Normalize rotations to [-π, π]
    gt_yaw = ((gt_tensor[0, 6] + np.pi) % (2 * np.pi)) - np.pi
    pred_yaw = ((pred_tensor[0, 6] + np.pi) % (2 * np.pi)) - np.pi
    gt_tensor[0, 6] = gt_yaw
    pred_tensor[0, 6] = pred_yaw

    # Validate positive dimensions
    if np.any(gt_tensor[0, 3:6] < 1e-6) or np.any(pred_tensor[0, 3:6] < 1e-6):
        return 0.0

    # Convert to LiDARInstance3DBoxes
    gt_tensor = torch.from_numpy(gt_tensor)
    pred_tensor = torch.from_numpy(pred_tensor)

    gt_boxes = LiDARInstance3DBoxes(gt_tensor, box_dim=7, origin=(0.5, 0.5, 0.5))
    pred_boxes = LiDARInstance3DBoxes(pred_tensor, box_dim=7, origin=(0.5, 0.5, 0.5))

    # Get BEV representation and compute IoU
    gt_bev = gt_boxes.bev.cpu()
    pred_bev = pred_boxes.bev.cpu()

    # Clamp to avoid numerical issues
    gt_bev[:, 2:4] = gt_bev[:, 2:4].clamp(min=1e-4)
    pred_bev[:, 2:4] = pred_bev[:, 2:4].clamp(min=1e-4)

    try:
        iou_tensor = box_iou_rotated(gt_bev, pred_bev)
        iou = iou_tensor.cpu().numpy()[0, 0]
        if np.isnan(iou) or iou < 0 or iou > 1:
            return 0.0
        return float(iou)
    except Exception as e:
        warnings.warn(f"BEV IoU calculation failed: {e}")
        return 0.0


def iou_3d(gt_box: dict, pred_box: dict) -> float:
    """Compute 3D IoU between two 3D boxes.

    Uses mmdet3d's official LiDARInstance3DBoxes.overlaps for accurate 3D IoU.

    Args:
        gt_box: GT box dict with 'translation', 'size'/'wlh', 'rotation' keys.
        pred_box: Predicted box dict with same structure.

    Returns:
        3D IoU value in [0, 1].
    """
    # Handle both "size" and "wlh" field names
    gt_size = gt_box.get("size", gt_box.get("wlh"))
    pred_size = pred_box.get("size", pred_box.get("wlh"))

    if gt_size is None or pred_size is None:
        return 0.0

    # Build tensors: [x, y, z, dx, dy, dz, yaw]
    gt_tensor = np.array([[
        gt_box["translation"][0],
        gt_box["translation"][1],
        gt_box["translation"][2],
        gt_size[0],
        gt_size[1],
        gt_size[2],
        gt_box["rotation"]
    ]], dtype=np.float32)

    pred_tensor = np.array([[
        pred_box["translation"][0],
        pred_box["translation"][1],
        pred_box["translation"][2],
        pred_size[0],
        pred_size[1],
        pred_size[2],
        pred_box["rotation"]
    ]], dtype=np.float32)

    # Normalize rotations to [-π, π]
    gt_yaw = ((gt_tensor[0, 6] + np.pi) % (2 * np.pi)) - np.pi
    pred_yaw = ((pred_tensor[0, 6] + np.pi) % (2 * np.pi)) - np.pi
    gt_tensor[0, 6] = gt_yaw
    pred_tensor[0, 6] = pred_yaw

    # Validate positive dimensions
    if np.any(gt_tensor[0, 3:6] < 1e-6) or np.any(pred_tensor[0, 3:6] < 1e-6):
        return 0.0

    # Convert to LiDARInstance3DBoxes
    gt_tensor = torch.from_numpy(gt_tensor)
    pred_tensor = torch.from_numpy(pred_tensor)

    gt_boxes = LiDARInstance3DBoxes(gt_tensor, box_dim=7, origin=(0.5, 0.5, 0.5))
    pred_boxes = LiDARInstance3DBoxes(pred_tensor, box_dim=7, origin=(0.5, 0.5, 0.5))

    # Compute 3D IoU
    try:
        gt_boxes = gt_boxes.to('cpu')
        pred_boxes = pred_boxes.to('cpu')
        iou_tensor = LiDARInstance3DBoxes.overlaps(gt_boxes, pred_boxes, mode='iou')
        iou = iou_tensor.cpu().numpy()[0, 0]
        if np.isnan(iou) or iou < 0 or iou > 1:
            return 0.0
        return float(iou)
    except Exception as e:
        warnings.warn(f"3D IoU calculation failed: {e}")
        return 0.0


# =============================================================================
# TP error metric functions (for NuScenes-style evaluation)
# =============================================================================

def velocity_l2(gt_box: dict, pred_box: dict) -> float:
    """Compute L2 distance between velocity vectors (XY only).

    Args:
        gt_box: GT box dict with 'velocity' key.
        pred_box: Predicted box dict with 'velocity' key.

    Returns:
        L2 distance in m/s.
    """
    return np.linalg.norm(np.array(pred_box["velocity"]) - np.array(gt_box["velocity"]))


def yaw_diff(gt_box: dict, pred_box: dict, period: float = 2 * np.pi) -> float:
    """Compute yaw angle difference between two boxes.

    Args:
        gt_box: GT box dict with 'rotation' key (yaw in radians).
        pred_box: Predicted box dict with 'rotation' key.
        period: Periodicity for angle difference (default: 2π).

    Returns:
        Absolute yaw difference in radians, in range [0, π].
    """
    yaw_gt = gt_box["rotation"]
    yaw_pred = pred_box["rotation"]

    # Compute smallest angle difference
    diff = (yaw_pred - yaw_gt + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)

    return abs(diff)


def scale_iou(gt_box: dict, pred_box: dict) -> float:
    """Compute scale IoU (box size similarity).

    Equivalent to 3D IoU if boxes are perfectly aligned (translation and rotation identical).

    Args:
        gt_box: GT box dict with 'size' key [w, l, h].
        pred_box: Predicted box dict with 'size' key.

    Returns:
        Scale IoU in [0, 1].
    """
    gt_size = np.array(gt_box["size"])
    pred_size = np.array(pred_box["size"])

    assert np.all(gt_size > 0), 'GT sizes must be > 0'
    assert np.all(pred_size > 0), 'Predicted sizes must be > 0'

    min_wlh = np.minimum(gt_size, pred_size)
    intersection = np.prod(min_wlh)
    union = np.prod(gt_size) + np.prod(pred_size) - intersection

    return float(intersection / union)


# =============================================================================
# Filtering and difficulty functions
# =============================================================================

def get_difficulty_level(box: dict, difficulty: str = 'all') -> bool:
    """Check if box matches KITTI-style difficulty level.

    TUMTraf doesn't have occlusion/truncation labels like KITTI, so we use:
    - num_pts (LiDAR points): proxy for visibility/occlusion
    - ego_dist (distance from ego): proxy for truncation

    Difficulty thresholds:
    - Easy: points >= 10, distance <= 40m
    - Moderate: 5 <= points < 10 OR (points >= 10 and 40m < dist <= 60m)
    - Hard: points < 5 OR dist > 60m (but still <= 80m)

    Args:
        box: Box dict with 'num_pts', 'ego_dist' keys.
        difficulty: 'easy', 'moderate', 'hard', or 'all'.

    Returns:
        True if box matches the difficulty level.
    """
    if difficulty == 'all':
        return True

    num_pts = box.get('num_pts', 0)
    ego_dist = box.get('ego_dist', float('inf'))

    if difficulty == 'easy':
        return num_pts >= 10 and ego_dist <= 40.0
    elif difficulty == 'moderate':
        return (5 <= num_pts < 10 or (num_pts >= 10 and 40.0 < ego_dist <= 60.0)) and ego_dist <= 60.0
    elif difficulty == 'hard':
        return (num_pts < 5 or ego_dist > 60.0) and num_pts >= 1 and ego_dist <= 80.0
    else:
        return True


def filter_eval_boxes(
    eval_boxes: Dict[str, List[dict]],
    max_dist: Dict[str, float],
    verbose: bool = False
) -> Dict[str, List[dict]]:
    """Filter boxes by distance range and LiDAR points.

    Args:
        eval_boxes: Dict mapping timestamp to list of boxes.
        max_dist: Dict mapping class name to max evaluation distance.
        verbose: Whether to print filtering statistics.

    Returns:
        Filtered eval_boxes dict.
    """
    total, dist_filter, point_filter = 0, 0, 0

    for timestamp in eval_boxes:
        total += len(eval_boxes[timestamp])

        # Filter by distance and class range
        eval_boxes[timestamp] = [
            box for box in eval_boxes[timestamp]
            if box["detection_name"] in max_dist and box["ego_dist"] < max_dist[box["detection_name"]]
        ]
        dist_filter += len(eval_boxes[timestamp])

        # Remove boxes with zero LiDAR points
        eval_boxes[timestamp] = [
            box for box in eval_boxes[timestamp] if box["num_pts"] != 0
        ]
        point_filter += len(eval_boxes[timestamp])

    if verbose:
        print(f"=> Original number of boxes: {total}")
        print(f"=> After distance filtering: {dist_filter}")
        print(f"=> After LiDAR points filtering: {point_filter}")

    return eval_boxes


# =============================================================================
# Accumulation functions (matching and collecting TP/FP)
# =============================================================================

def accumulate_center_distance(
    gt_boxes: Dict[str, List[dict]],
    pred_boxes: Dict[str, List[dict]],
    class_name: str,
    dist_th: float,
    verbose: bool = False
) -> dict:
    """Accumulate TP/FP using center distance matching (NuScenes-style).

    Args:
        gt_boxes: Dict mapping timestamp to list of GT boxes.
        pred_boxes: Dict mapping timestamp to list of predicted boxes.
        class_name: Class name to evaluate.
        dist_th: Distance threshold for matching (meters).
        verbose: Whether to print debug info.

    Returns:
        Dict with arrays: recall, precision, confidence, trans_err, vel_err, scale_err, orient_err.
    """
    # Count positives
    gt_boxes_all = []
    for boxes in gt_boxes.values():
        gt_boxes_all.extend(boxes)
    npos = len([1 for box in gt_boxes_all if box["detection_name"] == class_name])

    if verbose:
        print(f"Found {npos} GT of class {class_name} for distance {dist_th}m")

    # No GT for this class
    if npos == 0:
        return {
            "recall": np.linspace(0, 1, 101),
            "precision": np.zeros(101),
            "confidence": np.zeros(101),
            "trans_err": np.ones(101),
            "vel_err": np.ones(101),
            "scale_err": np.ones(101),
            "orient_err": np.ones(101)
        }

    # Collect predictions for this class
    pred_boxes_all = []
    for boxes in pred_boxes.values():
        pred_boxes_all.extend(boxes)
    pred_boxes_list = [box for box in pred_boxes_all if box["detection_name"] == class_name]
    pred_confs = [box["detection_score"] for box in pred_boxes_list]

    if verbose:
        print(f"Found {len(pred_confs)} predictions of class {class_name}")

    # Sort by confidence descending
    sortind = np.argsort(pred_confs)[::-1]

    # Match predictions to GT
    tp, fp, conf = [], [], []
    trans_err, vel_err, scale_err, orient_err = [], [], [], []

    taken = set()  # Track matched GT boxes

    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        # Find closest GT box
        for gt_idx, gt_box in enumerate(gt_boxes[pred_box["timestamp"]]):
            if (gt_box["detection_name"] == class_name and
                (pred_box["timestamp"], gt_idx) not in taken):
                this_dist = center_distance(gt_box, pred_box)
                if this_dist < min_dist:
                    min_dist = this_dist
                    match_gt_idx = gt_idx

        # Check if match
        is_match = min_dist < dist_th

        if is_match:
            taken.add((pred_box["timestamp"], match_gt_idx))
            tp.append(1)
            fp.append(0)
            conf.append(pred_box["detection_score"])

            # Compute TP errors
            gt_box_match = gt_boxes[pred_box["timestamp"]][match_gt_idx]
            trans_err.append(center_distance(gt_box_match, pred_box))
            vel_err.append(velocity_l2(gt_box_match, pred_box))
            scale_err.append(1 - scale_iou(gt_box_match, pred_box))

            # Barrier orientation only determined up to 180°
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            orient_err.append(yaw_diff(gt_box_match, pred_box, period=period))
        else:
            # False positive
            tp.append(0)
            fp.append(1)
            conf.append(pred_box["detection_score"])
            trans_err.append(np.nan)
            vel_err.append(np.nan)
            scale_err.append(np.nan)
            orient_err.append(np.nan)

    # No matches
    if len(trans_err) == 0 or all(np.isnan(trans_err)):
        return {
            "recall": np.linspace(0, 1, 101),
            "precision": np.zeros(101),
            "confidence": np.zeros(101),
            "trans_err": np.ones(101),
            "vel_err": np.ones(101),
            "scale_err": np.ones(101),
            "orient_err": np.ones(101)
        }

    # Compute precision and recall
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf_arr = np.array(conf)

    prec = tp / (fp + tp)
    rec = tp / float(npos)

    # Interpolate to 101 points
    rec_interp = np.linspace(0, 1, 101)
    prec_interp = np.interp(rec_interp, rec, prec, right=0)
    conf_interp = np.interp(rec_interp, rec, conf_arr, right=0)

    # Compute cumulative mean of errors (NaN-aware)
    def cummean_nan_aware(x):
        """Cumulative mean ignoring NaN values."""
        if np.all(np.isnan(x)):
            return np.ones(len(x))
        sum_vals = np.nancumsum(x.astype(float))
        count_vals = np.cumsum(~np.isnan(x))
        result = np.divide(sum_vals, count_vals, out=np.ones_like(sum_vals), where=count_vals != 0)
        result[count_vals == 0] = 1.0
        return result

    trans_err_cum = cummean_nan_aware(np.array(trans_err))
    vel_err_cum = cummean_nan_aware(np.array(vel_err))
    scale_err_cum = cummean_nan_aware(np.array(scale_err))
    orient_err_cum = cummean_nan_aware(np.array(orient_err))

    # Handle duplicate recall values for interpolation
    unique_indices = []
    seen_rec = set()
    for i in range(len(rec) - 1, -1, -1):
        if rec[i] not in seen_rec:
            unique_indices.insert(0, i)
            seen_rec.add(rec[i])

    rec_unique = rec[unique_indices]
    trans_err_interp = np.interp(rec_interp, rec_unique, trans_err_cum[unique_indices], left=1.0, right=1.0)
    vel_err_interp = np.interp(rec_interp, rec_unique, vel_err_cum[unique_indices], left=1.0, right=1.0)
    scale_err_interp = np.interp(rec_interp, rec_unique, scale_err_cum[unique_indices], left=1.0, right=1.0)
    orient_err_interp = np.interp(rec_interp, rec_unique, orient_err_cum[unique_indices], left=1.0, right=1.0)

    return {
        "recall": rec_interp,
        "precision": prec_interp,
        "confidence": conf_interp,
        "trans_err": trans_err_interp,
        "vel_err": vel_err_interp,
        "scale_err": scale_err_interp,
        "orient_err": orient_err_interp
    }


def accumulate_iou(
    gt_boxes: Dict[str, List[dict]],
    pred_boxes: Dict[str, List[dict]],
    class_name: str,
    iou_th: float,
    iou_type: str = 'bev',
    difficulty: str = 'all',
    verbose: bool = False
) -> dict:
    """Accumulate TP/FP using IoU matching (KITTI-style).

    Args:
        gt_boxes: Dict mapping timestamp to list of GT boxes.
        pred_boxes: Dict mapping timestamp to list of predicted boxes.
        class_name: Class name to evaluate.
        iou_th: IoU threshold for matching (e.g., 0.5, 0.7).
        iou_type: 'bev' for BEV mAP or '3d' for 3D mAP.
        difficulty: Difficulty level ('easy', 'moderate', 'hard', or 'all').
        verbose: Whether to print debug info.

    Returns:
        Dict with arrays: recall, precision, confidence.
    """
    # Count positives (filtered by difficulty)
    gt_boxes_all = []
    for boxes in gt_boxes.values():
        gt_boxes_all.extend(boxes)
    npos = len([
        1 for box in gt_boxes_all
        if box["detection_name"] == class_name and get_difficulty_level(box, difficulty)
    ])

    if verbose:
        print(f"Found {npos} GT of class {class_name} for {iou_type.upper()} IoU @ {iou_th}, difficulty={difficulty}")

    # No GT for this class
    if npos == 0:
        return {
            "recall": np.linspace(0, 1, 101),
            "precision": np.zeros(101),
            "confidence": np.zeros(101),
        }

    # Collect predictions for this class
    pred_boxes_all = []
    for boxes in pred_boxes.values():
        pred_boxes_all.extend(boxes)
    pred_boxes_list = [box for box in pred_boxes_all if box["detection_name"] == class_name]
    pred_confs = [box["detection_score"] for box in pred_boxes_list]

    if verbose:
        print(f"Found {len(pred_confs)} predictions of class {class_name}")

    # Sort by confidence descending
    sortind = np.argsort(pred_confs)[::-1]

    # Match predictions to GT
    tp, fp, conf = [], [], []
    taken = set()

    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        max_iou = 0.0
        match_gt_idx = None

        # Find best matching GT box (filtered by difficulty)
        for gt_idx, gt_box in enumerate(gt_boxes[pred_box["timestamp"]]):
            if (gt_box["detection_name"] == class_name and
                (pred_box["timestamp"], gt_idx) not in taken and
                get_difficulty_level(gt_box, difficulty)):

                # Compute IoU
                if iou_type == 'bev':
                    this_iou = bev_iou(gt_box, pred_box)
                elif iou_type == '3d':
                    this_iou = iou_3d(gt_box, pred_box)
                else:
                    raise ValueError(f"Unknown iou_type: {iou_type}")

                if this_iou > max_iou:
                    max_iou = this_iou
                    match_gt_idx = gt_idx

        # Check if match
        is_match = max_iou >= iou_th

        if is_match:
            taken.add((pred_box["timestamp"], match_gt_idx))
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
        conf.append(pred_box["detection_score"])

    # No matches
    if len(tp) == 0:
        return {
            "recall": np.linspace(0, 1, 101),
            "precision": np.zeros(101),
            "confidence": np.zeros(101),
        }

    # Compute precision and recall
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf_arr = np.array(conf)

    prec = tp / (fp + tp)
    rec = tp / float(npos)

    # Interpolate to 101 points
    rec_interp = np.linspace(0, 1, 101)
    prec_interp = np.interp(rec_interp, rec, prec, right=0)
    conf_interp = np.interp(rec_interp, rec, conf_arr, right=0)

    return {
        "recall": rec_interp,
        "precision": prec_interp,
        "confidence": conf_interp,
    }


# =============================================================================
# AP and TP error calculation
# =============================================================================

def calc_ap(metric_data: dict, min_recall: float, min_precision: float) -> float:
    """Calculate Average Precision from metric data.

    Args:
        metric_data: Dict with 'precision' array (101 points).
        min_recall: Minimum recall threshold (e.g., 0.1).
        min_precision: Minimum precision threshold (e.g., 0.1).

    Returns:
        Average Precision value.
    """
    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(metric_data["precision"])
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0

    return float(np.mean(prec)) / (1.0 - min_precision)


def calc_tp_errors(
    metric_data: dict,
    min_recall: float,
    metric_name: str
) -> float:
    """Calculate mean TP error from metric data.

    Args:
        metric_data: Dict with error arrays and 'confidence'.
        min_recall: Minimum recall threshold.
        metric_name: Error metric name ('trans_err', 'vel_err', etc.).

    Returns:
        Mean TP error value.
    """
    first_ind = round(100 * min_recall) + 1

    # Find max achieved recall (last non-zero confidence)
    non_zero = np.nonzero(metric_data["confidence"])[0]
    if len(non_zero) == 0:
        return 1.0

    last_ind = non_zero[-1]

    if last_ind < first_ind:
        return 1.0

    return float(np.mean(metric_data[metric_name][first_ind:last_ind + 1]))
