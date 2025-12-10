#!/usr/bin/env python3
"""Inspect checkpoint files to understand weight structure."""

import torch
import sys
from pathlib import Path

def inspect_checkpoint(checkpoint_path, name):
    """Inspect a checkpoint file and print key statistics."""
    print(f"\n{'='*80}")
    print(f"Inspecting: {name}")
    print(f"Path: {checkpoint_path}")
    print(f"{'='*80}")
    
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check checkpoint structure
        print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"\nState dict has {len(state_dict)} keys")
            
            # Categorize keys
            camera_backbone_keys = []
            camera_neck_keys = []
            camera_vtransform_keys = []
            lidar_keys = []
            fusion_keys = []
            head_keys = []
            other_keys = []
            
            for key in state_dict.keys():
                if 'camera.backbone' in key:
                    camera_backbone_keys.append(key)
                elif 'camera.neck' in key:
                    camera_neck_keys.append(key)
                elif 'camera.vtransform' in key:
                    camera_vtransform_keys.append(key)
                elif 'lidar' in key:
                    lidar_keys.append(key)
                elif 'fuser' in key or 'fusion' in key:
                    fusion_keys.append(key)
                elif 'head' in key or 'bbox_head' in key:
                    head_keys.append(key)
                else:
                    other_keys.append(key)
            
            print(f"\nKey categories:")
            print(f"  Camera backbone: {len(camera_backbone_keys)} keys")
            print(f"  Camera neck: {len(camera_neck_keys)} keys")
            print(f"  Camera vtransform: {len(camera_vtransform_keys)} keys")
            print(f"  LiDAR: {len(lidar_keys)} keys")
            print(f"  Fusion: {len(fusion_keys)} keys")
            print(f"  Head: {len(head_keys)} keys")
            print(f"  Other: {len(other_keys)} keys")
            
            # Show sample keys
            if camera_backbone_keys:
                print(f"\nSample camera backbone keys (first 5):")
                for key in camera_backbone_keys[:5]:
                    print(f"  {key}")
            
            if camera_neck_keys:
                print(f"\nSample camera neck keys (first 3):")
                for key in camera_neck_keys[:3]:
                    print(f"  {key}")
            
            if camera_vtransform_keys:
                print(f"\nSample camera vtransform keys (first 3):")
                for key in camera_vtransform_keys[:3]:
                    print(f"  {key}")
            
            # Check for vehicle/infrastructure prefixes
            vehicle_keys = [k for k in state_dict.keys() if 'vehicle' in k]
            infra_keys = [k for k in state_dict.keys() if 'infrastructure' in k]
            print(f"\nVehicle keys: {len(vehicle_keys)}")
            print(f"Infrastructure keys: {len(infra_keys)}")
            
            # Show first few keys to understand structure
            print(f"\nFirst 10 keys in checkpoint:")
            for i, key in enumerate(list(state_dict.keys())[:10]):
                shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
                print(f"  {i+1}. {key} -> shape: {shape}")
            
            return state_dict
        else:
            print("WARNING: No 'state_dict' key in checkpoint")
            return None
            
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    weights_dir = Path(__file__).parent.parent / "weights"
    
    # Inspect YOLOv8 checkpoint
    yolov8_path = weights_dir / "yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1_new.pth"
    yolov8_state_dict = inspect_checkpoint(yolov8_path, "YOLOv8 Checkpoint")
    
    # Inspect main model checkpoint
    main_path = weights_dir / "coopdet3d_vi_l_pointpillars512_2xtestgrid.pth"
    main_state_dict = inspect_checkpoint(main_path, "Main Model Checkpoint")
    
    # Compare keys
    if yolov8_state_dict and main_state_dict:
        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}")
        
        # Find camera backbone keys in YOLOv8
        yolov8_camera_keys = [k for k in yolov8_state_dict.keys() if 'backbone' in k]
        print(f"\nYOLOv8 camera backbone keys: {len(yolov8_camera_keys)}")
        if yolov8_camera_keys:
            print("Sample YOLOv8 keys (first 5):")
            for key in yolov8_camera_keys[:5]:
                print(f"  {key}")
        
        # Find camera backbone keys expected in main checkpoint
        main_camera_keys = [k for k in main_state_dict.keys() if 'camera.backbone' in k]
        print(f"\nMain checkpoint camera backbone keys: {len(main_camera_keys)}")
        if main_camera_keys:
            print("Sample main checkpoint camera keys (first 5):")
            for key in main_camera_keys[:5]:
                print(f"  {key}")
        
        # Check if YOLOv8 keys can be mapped to main checkpoint keys
        print(f"\n{'='*80}")
        print("KEY MAPPING ANALYSIS")
        print(f"{'='*80}")
        
        # YOLOv8 keys should be like: "backbone.stem.conv.weight"
        # Main checkpoint expects: "nodes.vehicle.fusion_model.encoders.camera.backbone.stem.conv.weight"
        if yolov8_camera_keys and main_camera_keys:
            sample_yolo = yolov8_camera_keys[0]
            sample_main = main_camera_keys[0]
            print(f"\nYOLOv8 key format: {sample_yolo}")
            print(f"Main checkpoint key format: {sample_main}")
            
            # Extract backbone part from main checkpoint
            if 'backbone' in sample_main:
                backbone_part = sample_main.split('backbone', 1)[1]
                print(f"Backbone part in main checkpoint: {backbone_part}")
            
            # Extract backbone part from YOLOv8
            if 'backbone' in sample_yolo:
                backbone_part_yolo = sample_yolo.split('backbone', 1)[1] if 'backbone' in sample_yolo else sample_yolo
                print(f"Backbone part in YOLOv8: {backbone_part_yolo}")

if __name__ == "__main__":
    main()

