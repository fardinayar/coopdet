#!/usr/bin/env python3
"""Inspect model structure to see what weight names it expects."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mmengine import Config
from coopdet3d.models import build_coop_model

def main():
    # Load a config file (use a default one)
    config_path = project_root / "configs" / "pointpillars_yolov8" / "pointpillars_yolov8.py"
    
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        print("Trying to find any config file...")
        config_files = list((project_root / "configs").rglob("*.py"))
        if config_files:
            config_path = config_files[0]
            print(f"Using: {config_path}")
        else:
            print("ERROR: No config file found")
            return
    
    cfg = Config.fromfile(str(config_path))
    
    # Build model
    print("Building model...")
    model = build_coop_model(cfg.model)
    
    # Get all parameter names
    print("\n" + "="*80)
    print("MODEL PARAMETER NAMES")
    print("="*80)
    
    all_params = dict(model.named_parameters())
    all_buffers = dict(model.named_buffers())
    
    # Categorize
    camera_backbone_params = []
    camera_neck_params = []
    camera_vtransform_params = []
    lidar_params = []
    fusion_params = []
    head_params = []
    other_params = []
    
    for name in list(all_params.keys()) + list(all_buffers.keys()):
        if 'camera.backbone' in name:
            camera_backbone_params.append(name)
        elif 'camera.neck' in name:
            camera_neck_params.append(name)
        elif 'camera.vtransform' in name:
            camera_vtransform_params.append(name)
        elif 'lidar' in name:
            lidar_params.append(name)
        elif 'fuser' in name or 'fusion' in name:
            fusion_params.append(name)
        elif 'head' in name or 'bbox_head' in name:
            head_params.append(name)
        else:
            other_params.append(name)
    
    print(f"\nTotal parameters: {len(all_params)}")
    print(f"Total buffers: {len(all_buffers)}")
    print(f"\nCategories:")
    print(f"  Camera backbone: {len(camera_backbone_params)}")
    print(f"  Camera neck: {len(camera_neck_params)}")
    print(f"  Camera vtransform: {len(camera_vtransform_params)}")
    print(f"  LiDAR: {len(lidar_params)}")
    print(f"  Fusion: {len(fusion_params)}")
    print(f"  Head: {len(head_params)}")
    print(f"  Other: {len(other_params)}")
    
    if camera_backbone_params:
        print(f"\nSample camera backbone parameter names (first 10):")
        for name in camera_backbone_params[:10]:
            print(f"  {name}")
    
    if camera_neck_params:
        print(f"\nSample camera neck parameter names (first 5):")
        for name in camera_neck_params[:5]:
            print(f"  {name}")
    
    if camera_vtransform_params:
        print(f"\nSample camera vtransform parameter names (first 5):")
        for name in camera_vtransform_params[:5]:
            print(f"  {name}")
    
    # Check vehicle vs infrastructure
    vehicle_params = [p for p in all_params.keys() if 'vehicle' in p]
    infra_params = [p for p in all_params.keys() if 'infrastructure' in p]
    print(f"\nVehicle parameters: {len(vehicle_params)}")
    print(f"Infrastructure parameters: {len(infra_params)}")
    
    # Show structure for camera backbone
    if camera_backbone_params:
        print(f"\n{'='*80}")
        print("CAMERA BACKBONE STRUCTURE")
        print(f"{'='*80}")
        # Extract the part after "backbone."
        backbone_parts = set()
        for name in camera_backbone_params:
            if 'backbone.' in name:
                part = name.split('backbone.', 1)[1]
                backbone_parts.add(part.split('.')[0])  # First component
        
        print(f"Backbone components: {sorted(backbone_parts)}")
        
        # Show full path for vehicle camera backbone
        vehicle_camera_backbone = [p for p in camera_backbone_params if 'vehicle' in p]
        if vehicle_camera_backbone:
            print(f"\nFull vehicle camera backbone path (first key):")
            print(f"  {vehicle_camera_backbone[0]}")
            # Extract what comes after "backbone."
            if 'backbone.' in vehicle_camera_backbone[0]:
                backbone_suffix = vehicle_camera_backbone[0].split('backbone.', 1)[1]
                print(f"  Backbone suffix: {backbone_suffix}")

if __name__ == "__main__":
    main()

