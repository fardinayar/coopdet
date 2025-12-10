#!/usr/bin/env python3
"""Debug script to check IoU calculation between GT and predictions."""

import pickle
import numpy as np
import torch
from mmdet3d.structures import LiDARInstance3DBoxes

# Load GT from pkl
with open('data/tumtraf_v2x_cooperative_perception_dataset_processed/tumtraf_v2x_nusc_infos_val.pkl', 'rb') as f:
    val_data = pickle.load(f)

# Get first sample with a BUS
for sample in val_data['infos'][:10]:
    for i, (box, name) in enumerate(zip(sample['gt_boxes'], sample['gt_names'])):
        if name == 'BUS':
            print("="*70)
            print(f"GT BUS box from pkl:")
            print(f"  Center (x,y,z): [{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}]")
            print(f"  Dimensions: [{box[3]:.2f}, {box[4]:.2f}, {box[5]:.2f}]")
            print(f"  Yaw: {box[6]:.4f}")

            # Create LiDARInstance3DBoxes
            gt_tensor = torch.tensor([[box[0], box[1], box[2], box[3], box[4], box[5], box[6]]], dtype=torch.float32)
            gt_boxes = LiDARInstance3DBoxes(gt_tensor, box_dim=7, origin=(0.5, 0.5, 0.5))

            print(f"\nLiDARInstance3DBoxes dims property:")
            print(f"  dims: {gt_boxes.dims}")
            print(f"  dims[0, 0] (should be length=12.2): {gt_boxes.dims[0, 0]:.2f}")
            print(f"  dims[0, 1] (should be width=2.87): {gt_boxes.dims[0, 1]:.2f}")
            print(f"  dims[0, 2] (should be height=3.22): {gt_boxes.dims[0, 2]:.2f}")

            # Create a "perfect" prediction (slightly offset for testing)
            pred_tensor = torch.tensor([[box[0]+0.1, box[1]+0.1, box[2], box[3], box[4], box[5], box[6]]], dtype=torch.float32)
            pred_boxes = LiDARInstance3DBoxes(pred_tensor, box_dim=7, origin=(0.5, 0.5, 0.5))

            # Compute BEV IoU
            try:
                from mmcv.ops import box_iou_rotated
            except ImportError:
                from mmdet.structures.bbox import box_iou_rotated

            gt_bev = gt_boxes.bev.cpu()
            pred_bev = pred_boxes.bev.cpu()
            bev_iou = box_iou_rotated(gt_bev, pred_bev).cpu().numpy()[0, 0]

            # Compute 3D IoU
            iou_3d = LiDARInstance3DBoxes.overlaps(gt_boxes.to('cpu'), pred_boxes.to('cpu'), mode='iou').cpu().numpy()[0, 0]

            print(f"\nIoU test (prediction offset by 0.1m in x,y):")
            print(f"  BEV IoU: {bev_iou:.4f} (should be ~0.9+)")
            print(f"  3D IoU: {iou_3d:.4f} (should be ~0.9+)")

            # Now test with swapped dimensions
            pred_tensor_swapped = torch.tensor([[box[0]+0.1, box[1]+0.1, box[2], box[4], box[3], box[5], box[6]]], dtype=torch.float32)
            pred_boxes_swapped = LiDARInstance3DBoxes(pred_tensor_swapped, box_dim=7, origin=(0.5, 0.5, 0.5))

            gt_bev_test = gt_boxes.bev.cpu()
            pred_bev_swapped = pred_boxes_swapped.bev.cpu()
            bev_iou_swapped = box_iou_rotated(gt_bev_test, pred_bev_swapped).cpu().numpy()[0, 0]

            iou_3d_swapped = LiDARInstance3DBoxes.overlaps(gt_boxes.to('cpu'), pred_boxes_swapped.to('cpu'), mode='iou').cpu().numpy()[0, 0]

            print(f"\nIoU test with SWAPPED dimensions (l↔w):")
            print(f"  BEV IoU: {bev_iou_swapped:.4f} (should be much lower)")
            print(f"  3D IoU: {iou_3d_swapped:.4f} (should be much lower)")
            print("="*70)
            break
    else:
        continue
    break
