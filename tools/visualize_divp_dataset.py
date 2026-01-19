# Copyright (c) OpenMMLab. All rights reserved.
"""
DIVPデータセット可視化スクリプト

指定したフレームを抜粋して点群とbounding boxを可視化します。
"""
import argparse
import copy
import os
import random
import sys
from os import path as osp

import numpy as np
import torch
from mmengine import Config
from mmengine.utils import mkdir_or_exist
import trimesh

# Add package folder to pythonpath
base_dir = osp.dirname(osp.abspath(__file__))
coopdet_dir = osp.join(base_dir, '..')
sys.path.insert(0, coopdet_dir)

try:
    from mmdet3d.structures import LiDARInstance3DBoxes
except ImportError:
    try:
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
    except ImportError:
        raise ImportError("mmdet3d.structures not found. Please install mmdet3d.")

# Import build_dataset first - this will trigger pipeline imports via coopdet3d/datasets/__init__.py
# which registers all transforms in the registry
from coopdet3d.datasets import build_dataset

# Explicitly import pipelines to ensure transforms are registered
# This is a safety measure - build_dataset import should already trigger this
from coopdet3d.datasets import pipelines  # noqa: F401


# Color palette for object classes (matching visualization_hook.py style)
# Map class names to label indices for color lookup
CLASS_COLOR_MAP = {
    0: [255, 50, 50, 255],      # CAR - bright red
    1: [50, 255, 50, 255],      # TRAILER - bright green
    2: [50, 100, 255, 255],     # TRUCK - bright blue
    3: [255, 255, 50, 255],     # VAN - bright yellow
    4: [255, 50, 255, 255],     # PEDESTRIAN - bright magenta
    5: [50, 255, 255, 255],     # BUS - bright cyan
    6: [255, 150, 50, 255],     # BICYCLE - bright orange
}


def create_box_mesh(center, dims, rotation, color=[0, 255, 0, 255]):
    """Create a colored wireframe box mesh from center, dimensions, and rotation.
    
    Args:
        center: [x, y, z] center of box
        dims: [l, w, h] dimensions
        rotation: rotation angle in radians (yaw)
        color: RGBA color [0-255, 0-255, 0-255, 0-255]
    
    Returns:
        trimesh.Trimesh: Wireframe box as thin cylinders
    """
    l, w, h = dims
    
    # Create box vertices (8 corners)
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    corners = np.vstack([x_corners, y_corners, z_corners])  # 3 x 8
    
    # Rotate around z-axis (yaw)
    rot_mat = np.array([
        [np.cos(rotation), -np.sin(rotation), 0],
        [np.sin(rotation), np.cos(rotation), 0],
        [0, 0, 1]
    ])
    corners = rot_mat @ corners
    
    # Translate to center
    corners = corners.T + np.array(center)  # 8 x 3
    
    # Define edges connecting vertices
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
    ]
    
    # Create wireframe by creating thin cylinders for each edge
    meshes = []
    for edge in edges:
        p1 = corners[edge[0]]
        p2 = corners[edge[1]]
        # Create thin cylinder between points
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length > 1e-6:  # Avoid degenerate edges
            try:
                cylinder = trimesh.creation.cylinder(
                    radius=0.08,  # slightly thicker for better visibility
                    height=length,
                    sections=6
                )
                
                # Rotate and translate cylinder to connect p1 and p2
                direction_norm = direction / length
                
                # Create transformation matrix to align cylinder with edge
                # Cylinder default orientation is along Z axis
                z_axis = np.array([0, 0, 1])
                
                # Check if vectors are parallel or anti-parallel
                dot_product = np.clip(np.dot(z_axis, direction_norm), -1.0, 1.0)
                
                if abs(dot_product) < 0.9999:  # Not parallel
                    # Use cross product to find rotation axis
                    rotation_axis = np.cross(z_axis, direction_norm)
                    rotation_axis_norm = np.linalg.norm(rotation_axis)
                    if rotation_axis_norm > 1e-6:
                        rotation_axis = rotation_axis / rotation_axis_norm
                        rotation_angle = np.arccos(dot_product)
                        rotation_matrix = trimesh.transformations.rotation_matrix(
                            rotation_angle, rotation_axis
                        )
                        cylinder.apply_transform(rotation_matrix)
                elif dot_product < 0:  # Anti-parallel (pointing down)
                    # Rotate 180 degrees around X axis
                    rotation_matrix = trimesh.transformations.rotation_matrix(
                        np.pi, [1, 0, 0]
                    )
                    cylinder.apply_transform(rotation_matrix)
                # If parallel (dot_product > 0.9999), no rotation needed
                
                # Translate to position
                midpoint = (p1 + p2) / 2
                cylinder.apply_translation(midpoint)
                
                # Apply color to all faces
                cylinder.visual.face_colors = np.array(color, dtype=np.uint8)
                meshes.append(cylinder)
            except Exception as e:
                # Skip this edge if there's an error
                print(f"Warning: Failed to create cylinder for edge {edge}: {e}")
                continue
    
    # Combine all cylinders into one mesh
    if len(meshes) > 0:
        try:
            combined = trimesh.util.concatenate(meshes)
            return combined
        except Exception as e:
            print(f"Warning: Failed to concatenate meshes: {e}")
            return None
    return None


def boxes_to_mesh(boxes_tensor, labels, color_map=None):
    """Convert boxes to trimesh meshes.
    
    Args:
        boxes_tensor: Nx9 tensor [x, y, z, l, w, h, yaw, vx, vy]
        labels: N tensor of class labels
        color_map: dict mapping label -> [r, g, b, a]
    
    Returns:
        list of trimesh.Trimesh
    """
    if color_map is None:
        color_map = CLASS_COLOR_MAP
    
    meshes = []
    for i in range(len(boxes_tensor)):
        box = boxes_tensor[i].cpu().numpy() if hasattr(boxes_tensor[i], 'cpu') else boxes_tensor[i]
        label = int(labels[i].cpu().item()) if hasattr(labels[i], 'cpu') else int(labels[i])
        center = box[:3]
        dims = box[3:6]
        rotation = box[6]
        
        # 正解のbounding boxは緑色に統一
        color = [0, 255, 0, 255]  # 緑色
        
        mesh = create_box_mesh(center, dims, rotation, color)
        if mesh is not None:
            meshes.append(mesh)
    
    return meshes


def visualize_to_glb(
    fpath: str,
    vehicle_points: np.ndarray = None,
    infrastructure_points: np.ndarray = None,
    bboxes: LiDARInstance3DBoxes = None,
    labels: np.ndarray = None,
    classes: list = None,
) -> None:
    """Visualize two point clouds (vehicle and infrastructure) with bounding boxes as GLB file.
    
    Args:
        fpath: Output GLB file path
        vehicle_points: Vehicle point cloud (N, 4) [x, y, z, intensity]
        infrastructure_points: Infrastructure point cloud (M, 4) [x, y, z, intensity]
        bboxes: LiDARInstance3DBoxes object
        labels: Label array for bounding boxes
        classes: List of class names (for reference, not used directly)
    """
    # Create trimesh scene
    scene = trimesh.Scene()
    geometry_count = 0
    
    # Add vehicle point cloud (white)
    if vehicle_points is not None and len(vehicle_points) > 0:
        pc_xyz = vehicle_points[:, :3]
        # Downsample for visualization (max 50k points)
        # if len(pc_xyz) > 50000:
        #     indices = np.random.choice(len(pc_xyz), 50000, replace=False)
        #     pc_xyz = pc_xyz[indices]
        
        # Create point cloud (white for vehicle)
        pc_colors = np.ones((len(pc_xyz), 4), dtype=np.uint8) * [255, 255, 255, 255]
        pc_mesh = trimesh.points.PointCloud(pc_xyz, colors=pc_colors)
        scene.add_geometry(pc_mesh, node_name='vehicle_point_cloud')
        geometry_count += 1
        print(f'  Added vehicle point cloud with {len(pc_xyz)} points')
    
    # Add infrastructure point cloud (red)
    if infrastructure_points is not None and len(infrastructure_points) > 0:
        pc_xyz = infrastructure_points[:, :3]
        # Downsample for visualization (max 50k points)
        # if len(pc_xyz) > 50000:
        #     indices = np.random.choice(len(pc_xyz), 50000, replace=False)
        #     pc_xyz = pc_xyz[indices]
        
        # Create point cloud (red for infrastructure)
        pc_colors = np.ones((len(pc_xyz), 4), dtype=np.uint8) * [255, 0, 0, 255]
        pc_mesh = trimesh.points.PointCloud(pc_xyz, colors=pc_colors)
        scene.add_geometry(pc_mesh, node_name='infrastructure_point_cloud')
        geometry_count += 1
        print(f'  Added infrastructure point cloud with {len(pc_xyz)} points')
    
    # Add bounding boxes (colored by class)
    if bboxes is not None and len(bboxes) > 0 and labels is not None:
        print(f'  Processing {len(bboxes)} bounding boxes...')
        # Convert LiDARInstance3DBoxes to numpy array
        if isinstance(bboxes, LiDARInstance3DBoxes):
            boxes_tensor = bboxes.tensor.cpu().numpy()
            print(f'  Converted LiDARInstance3DBoxes to numpy, shape: {boxes_tensor.shape}')
        else:
            boxes_tensor = bboxes
            print(f'  Using bboxes as-is, shape: {boxes_tensor.shape}')
        
        # Convert labels to numpy if needed
        if isinstance(labels, torch.Tensor):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = labels
        print(f'  Labels shape: {labels_np.shape}')
        
        # Create meshes for boxes
        box_meshes = boxes_to_mesh(boxes_tensor, labels_np)
        print(f'  Created {len(box_meshes)} box meshes')
        for i, mesh in enumerate(box_meshes):
            if mesh is not None:
                scene.add_geometry(mesh, node_name=f'gt_box_{i}')
                geometry_count += 1
        added_count = len([m for m in box_meshes if m is not None])
        print(f'  Added {added_count} bounding boxes (colored by class)')
    else:
        print(f'  No bounding boxes to add (bboxes: {bboxes is not None}, len: {len(bboxes) if bboxes is not None else 0}, labels: {labels is not None})')
    
    print(f'  Total geometries in scene: {geometry_count}')
    
    # Save as GLB
    mkdir_or_exist(osp.dirname(fpath))
    
    # Validate scene before exporting
    if len(scene.geometry) == 0:
        print(f'Warning: Scene is empty, skipping export')
        return
    
    try:
        # Export with error handling
        scene.export(fpath, file_type='glb')
        print(f'  Saved GLB file: {fpath}')
        
        # Verify file was created and has size > 0
        if osp.exists(fpath) and osp.getsize(fpath) > 0:
            print(f'  GLB file verified: {osp.getsize(fpath)} bytes')
        else:
            print(f'  Error: GLB file not created or is empty: {fpath}')
    except Exception as e:
        print(f'  Error: Failed to export GLB file: {e}')
        import traceback
        traceback.print_exc()


def recursive_eval(obj, globals=None):
    """Recursively evaluate string expressions in config."""
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize DIVP dataset frames with point clouds and bounding boxes'
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to config file'
    )
    parser.add_argument(
        '--indices',
        type=str,
        default=None,
        help='Comma-separated list of frame indices to visualize (e.g., "0,10,20")'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of random frames to visualize (used when --indices is not specified)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to use'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default='viz_divp_dataset',
        help='Output directory for visualization images'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for frame selection'
    )
    args, opts = parser.parse_known_args()
    return args, opts


def main():
    """Main function."""
    args, opts = parse_args()

    # Load config
    try:
        from mmdet3d.utils import recursive_eval as mmdet3d_recursive_eval
        from torchpack.utils.config import configs
        configs.load(args.config, recursive=True)
        configs.update(opts)
        cfg_dict = mmdet3d_recursive_eval(configs)
        cfg = Config(cfg_dict, filename=args.config)
    except ImportError:
        # Fallback to simple config loading
        cfg = Config.fromfile(args.config)
        if opts:
            cfg.merge_from_dict(dict(opts))

    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Build dataset
    # Transforms should already be registered via the imports at the top of the file
    try:
        dataset = build_dataset(cfg.data[args.split])
    except KeyError as e:
        # Check if it's actually a KeyError for the split
        if args.split not in cfg.data:
            available_splits = [k for k in cfg.data.keys() if k not in ['samples_per_gpu', 'workers_per_gpu']]
            raise ValueError(
                f"Split '{args.split}' not found in config. "
                f"Available splits: {available_splits}"
            ) from e
        # If it's a different KeyError, re-raise it
        raise
    except Exception as e:
        # Check if it's a transform registration error
        error_str = str(e)
        if 'not in the' in error_str and 'registry' in error_str:
            raise RuntimeError(
                f"Transform registration error: {e}\n"
                "This usually means custom transforms were not imported before building the dataset.\n"
                "The pipelines should be automatically imported via 'from coopdet3d.datasets import build_dataset',\n"
                "but if this error persists, try explicitly importing: 'from coopdet3d.datasets import pipelines'"
            ) from e
        # Re-raise other exceptions as-is
        raise

    dataset_len = len(dataset)
    if dataset_len == 0:
        raise ValueError(f"Dataset is empty for split '{args.split}'")

    print(f"Dataset loaded: {dataset_len} frames in '{args.split}' split")

    # Determine frame indices
    if args.indices is not None:
        # Parse comma-separated indices
        try:
            indices = [int(idx.strip()) for idx in args.indices.split(',')]
        except ValueError:
            raise ValueError(
                f"Invalid indices format: '{args.indices}'. "
                "Expected comma-separated integers (e.g., '0,10,20')"
            )
        # Validate indices
        invalid_indices = [idx for idx in indices if idx < 0 or idx >= dataset_len]
        if invalid_indices:
            raise ValueError(
                f"Invalid indices: {invalid_indices}. "
                f"Valid range: [0, {dataset_len - 1}]"
            )
    else:
        # Random selection
        if args.num_samples > dataset_len:
            print(
                f"Warning: num-samples ({args.num_samples}) > dataset length ({dataset_len}). "
                f"Using all {dataset_len} frames."
            )
            indices = list(range(dataset_len))
        else:
            indices = random.sample(range(dataset_len), args.num_samples)
            indices.sort()  # Sort for easier viewing

    print(f"Visualizing {len(indices)} frame(s): {indices}")

    # Create output directory
    mkdir_or_exist(args.out_dir)

    # Get point cloud range from config
    point_cloud_range = cfg.get('point_cloud_range', [-225.0, -225.0, -8.0, 225.0, 225.0, 0.0])
    xlim = [point_cloud_range[0], point_cloud_range[3]]
    ylim = [point_cloud_range[1], point_cloud_range[4]]

    # Get object classes from config
    object_classes = cfg.get('object_classes', None)
    if object_classes is None:
        # Try to get from dataset
        try:
            object_classes = dataset.CLASSES
        except AttributeError:
            # Fallback to default
            object_classes = ['CAR', 'TRAILER', 'TRUCK', 'VAN', 'PEDESTRIAN', 'BUS', 'BICYCLE']

    print(f"Object classes: {object_classes}")

    # Visualize each frame
    for idx in indices:
        try:
            # Get data from dataset
            data = dataset[idx]

            # Extract point clouds
            # NOTE: The data from dataset[idx] has already been processed by the dataset pipeline.
            # This includes transformations such as:
            # - VehiclePointsToInfraCoords: Coordinate transformation from vehicle to infrastructure frame
            # - GlobalRotScaleTransCoop: Rotation, scaling, and translation (augmentation for train, fixed for val/test)
            # - ObjectRangeFilter: Filtering objects outside the range
            # This script does NOT apply any additional transformations - it only converts formats for visualization.
            
            # Data structure: data['inputs'] contains the actual data
            # vehicle_points and infrastructure_points are already torch.Tensor (not DataContainer)
            if 'inputs' in data:
                inputs = data['inputs']
                vehicle_points = inputs.get('vehicle_points', None)
                infrastructure_points = inputs.get('infrastructure_points', None)
            else:
                # Old format: direct keys
                vehicle_points = data.get('vehicle_points', None)
                infrastructure_points = data.get('infrastructure_points', None)

            # vehicle_points and infrastructure_points are already torch.Tensor
            # No need to extract from DataContainer - they are the actual tensors


            # Validate point cloud shapes
            if vehicle_points is None or infrastructure_points is None:
                print(f"Warning: Frame {idx} missing point cloud data. Skipping.")
                continue

            # Convert to numpy if needed (they are already torch.Tensor)
            # This is the ONLY transformation applied in this script - format conversion only
            if isinstance(vehicle_points, torch.Tensor):
                vehicle_points = vehicle_points.cpu().numpy()
            if isinstance(infrastructure_points, torch.Tensor):
                infrastructure_points = infrastructure_points.cpu().numpy()

            # Validate point cloud shapes
            if len(vehicle_points.shape) != 2 or vehicle_points.shape[0] == 0:
                print(f"Warning: Frame {idx} has invalid vehicle_points shape: {vehicle_points.shape}. Skipping.")
                continue
            if len(infrastructure_points.shape) != 2 or infrastructure_points.shape[0] == 0:
                print(f"Warning: Frame {idx} has invalid infrastructure_points shape: {infrastructure_points.shape}. Skipping.")
                continue

            # Extract only x, y, z, intensity (first 4 columns) for visualization
            # This is just column selection, not a coordinate transformation
            if vehicle_points.shape[1] >= 4:
                vehicle_points = vehicle_points[:, :4]
            else:
                print(f"Warning: Frame {idx} vehicle_points has insufficient columns: {vehicle_points.shape[1]}. Expected at least 4.")
            if infrastructure_points.shape[1] >= 4:
                infrastructure_points = infrastructure_points[:, :4]
            else:
                print(f"Warning: Frame {idx} infrastructure_points has insufficient columns: {infrastructure_points.shape[1]}. Expected at least 4.")

            # Extract bounding boxes and labels
            # NOTE: The bboxes from dataset[idx] have also been processed by the dataset pipeline.
            # They have been transformed along with the point clouds (coordinate transformation, augmentation, etc.)
            # This script does NOT apply any additional transformations - it only converts formats for visualization.
            
            bboxes = None
            labels = None
            timestamp = None
            
            # Debug: print all top-level keys in data
            print(f"  Frame {idx}: Top-level data keys: {list(data.keys())}")
            
            # Check both old format (direct keys) and new format (data_samples)
            if 'data_samples' in data:
                # New format: data_samples is Det3DDataSample object (not a list)
                data_sample = data['data_samples']
                print(f"  Frame {idx}: data_sample type: {type(data_sample)}")
                
                if hasattr(data_sample, 'gt_instances_3d'):
                    gt_instances = data_sample.gt_instances_3d
                    print(f"  Frame {idx}: gt_instances type: {type(gt_instances)}")
                    print(f"  Frame {idx}: gt_instances len: {len(gt_instances)}")
                    
                    # InstanceData supports dict-like access: keys(), get(), []
                    # First, check what keys are available
                    available_keys = list(gt_instances.keys())
                    print(f"  Frame {idx}: gt_instances keys: {available_keys}")
                    
                    # Try to get bboxes_3d and labels_3d using dict-like access
                    if 'bboxes_3d' in available_keys:
                        bboxes = gt_instances.get('bboxes_3d', None)
                        if bboxes is not None:
                            print(f"  Frame {idx}: Found bboxes_3d, type: {type(bboxes)}")
                    else:
                        # Try attribute access as fallback
                        bboxes = getattr(gt_instances, 'bboxes_3d', None)
                        if bboxes is not None:
                            print(f"  Frame {idx}: Found bboxes_3d via attribute access, type: {type(bboxes)}")
                    
                    if 'labels_3d' in available_keys:
                        labels = gt_instances.get('labels_3d', None)
                        if labels is not None:
                            print(f"  Frame {idx}: Found labels_3d, type: {type(labels)}")
                    else:
                        # Try attribute access as fallback
                        labels = getattr(gt_instances, 'labels_3d', None)
                        if labels is not None:
                            print(f"  Frame {idx}: Found labels_3d via attribute access, type: {type(labels)}")
                
                # Get timestamp from metainfo
                if hasattr(data_sample, 'metainfo') and 'timestamp' in data_sample.metainfo:
                    timestamp = data_sample.metainfo['timestamp']
            
            # Also check old format in case GT is stored differently
            if bboxes is None:
                if 'inputs' in data:
                    # Check if GT is in inputs (some formats put GT in inputs)
                    inputs = data['inputs']
                    print(f"  Frame {idx}: inputs keys: {list(inputs.keys())}")
                    if 'gt_bboxes_3d' in inputs:
                        bboxes = inputs['gt_bboxes_3d']
                        labels = inputs.get('gt_labels_3d', None)
                        print(f"  Frame {idx}: Found GT in inputs")
                elif 'gt_bboxes_3d' in data and 'gt_labels_3d' in data:
                    # Old format: direct keys
                    bboxes = data['gt_bboxes_3d']
                    labels = data['gt_labels_3d']
                    print(f"  Frame {idx}: Found GT in top-level data")
                    
                    # Handle DataContainer format
                    if hasattr(bboxes, 'data'):
                        bboxes = bboxes.data[0][0]
                    if hasattr(labels, 'data'):
                        labels = labels.data[0][0]

            # Convert to numpy if needed
            if bboxes is not None:
                print(f"  Frame {idx}: Converting bboxes from {type(bboxes)} to numpy")
                if isinstance(bboxes, LiDARInstance3DBoxes):
                    bboxes_np = bboxes.tensor.cpu().numpy()
                    print(f"  Frame {idx}: bboxes_np shape: {bboxes_np.shape}")
                    bboxes = bboxes_np
                elif isinstance(bboxes, torch.Tensor):
                    bboxes_np = bboxes.cpu().numpy()
                    print(f"  Frame {idx}: bboxes_np shape: {bboxes_np.shape}")
                    bboxes = bboxes_np
                else:
                    print(f"  Frame {idx}: bboxes is already numpy or other type: {type(bboxes)}")
            else:
                print(f"  Frame {idx}: bboxes is None - no bounding boxes found")

            if labels is not None and isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
                print(f"  Frame {idx}: labels converted to numpy, shape: {labels.shape}")

            # Convert to LiDARInstance3DBoxes for visualization
            if bboxes is not None and len(bboxes) > 0:
                print(f"  Frame {idx}: bboxes has {len(bboxes)} boxes")
                if bboxes.shape[1] >= 7:
                    # Ensure box_dim matches the data
                    box_dim = bboxes.shape[1]
                    print(f"  Frame {idx}: Creating LiDARInstance3DBoxes with box_dim={box_dim}")
                    bboxes = LiDARInstance3DBoxes(bboxes, box_dim=box_dim)
                    print(f"  Frame {idx}: LiDARInstance3DBoxes created successfully")
                else:
                    print(f"Warning: Frame {idx} has invalid bbox shape: {bboxes.shape}. Skipping bboxes.")
                    bboxes = None
                    labels = None
            else:
                # Empty bboxes
                print(f"  Frame {idx}: No bboxes to visualize (bboxes is None or empty)")
                bboxes = None
                labels = None

            # Get timestamp for filename (if not already extracted from data_samples)
            if timestamp is None:
                if 'metas' in data:
                    # Old format: metas
                    metas = data['metas']
                    # Handle DataContainer format
                    if hasattr(metas, 'data'):
                        metas = metas.data[0][0]
                    if isinstance(metas, (list, tuple)) and len(metas) > 0:
                        if isinstance(metas[0], dict):
                            timestamp = metas[0].get('timestamp', None)
                        elif isinstance(metas[0], (list, tuple)) and len(metas[0]) > 0:
                            if isinstance(metas[0][0], dict):
                                timestamp = metas[0][0].get('timestamp', None)
                    elif isinstance(metas, dict):
                        timestamp = metas.get('timestamp', None)
                elif 'timestamp' in data:
                    timestamp = data['timestamp']
                
                if timestamp is None:
                    timestamp = f"frame_{idx}"

            # Create output filename
            output_filename = osp.join(args.out_dir, f"frame_{idx}_{timestamp}.glb")

            # Extract file paths from metainfo
            vehicle_lidar_path = None
            infrastructure_lidar_path = None
            ann_file_path = None
            token = None
            
            if 'data_samples' in data:
                data_sample = data['data_samples']
                if hasattr(data_sample, 'metainfo'):
                    metainfo = data_sample.metainfo
                    vehicle_lidar_path = metainfo.get('vehicle_lidar_path', None)
                    infrastructure_lidar_path = metainfo.get('infrastructure_lidar_path', None)
                    ann_file_path = metainfo.get('ann_file', None)
                    token = metainfo.get('token', None)
            elif 'metas' in data:
                metas = data['metas']
                if hasattr(metas, 'data'):
                    metas = metas.data[0][0]
                if isinstance(metas, dict):
                    vehicle_lidar_path = metas.get('vehicle_lidar_path', None)
                    infrastructure_lidar_path = metas.get('infrastructure_lidar_path', None)
                    ann_file_path = metas.get('ann_file', None)
                    token = metas.get('token', None)
                elif isinstance(metas, (list, tuple)) and len(metas) > 0:
                    if isinstance(metas[0], dict):
                        vehicle_lidar_path = metas[0].get('vehicle_lidar_path', None)
                        infrastructure_lidar_path = metas[0].get('infrastructure_lidar_path', None)
                        ann_file_path = metas[0].get('ann_file', None)
                        token = metas[0].get('token', None)
            
            # Also try to get ann_file from dataset if available
            if ann_file_path is None and hasattr(dataset, 'ann_file'):
                ann_file_path = dataset.ann_file
            
            # Print file paths to stdout
            print(f"\n=== Frame {idx} Input Files ===")
            if vehicle_lidar_path:
                print(f"Vehicle LiDAR: {vehicle_lidar_path}")
            if infrastructure_lidar_path:
                print(f"Infrastructure LiDAR: {infrastructure_lidar_path}")
            if ann_file_path:
                print(f"Annotation file: {ann_file_path}")
            if token:
                print(f"Token: {token}")
            if not vehicle_lidar_path and not infrastructure_lidar_path:
                print("Warning: No file path information found in metainfo")
            print(f"Output GLB: {output_filename}")
            print("=" * 50)

            # Visualize
            print(f"Visualizing frame {idx} (timestamp: {timestamp})...")
            try:
                visualize_to_glb(
                    output_filename,
                    vehicle_points,
                    infrastructure_points,
                    bboxes=bboxes,
                    labels=labels,
                    classes=object_classes,
                )
            except Exception as e:
                print(f"  Error visualizing frame {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        except Exception as e:
            print(f"Error processing frame {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nVisualization complete! Output directory: {args.out_dir}")


if __name__ == '__main__':
    main()

