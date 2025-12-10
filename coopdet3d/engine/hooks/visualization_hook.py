"""Visualization hook for saving 3D predictions as GLB files."""
import os
import os.path as osp
import numpy as np
from typing import Sequence
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.fileio import get
from mmdet3d.registry import HOOKS
from mmdet3d.structures import Det3DDataSample
import trimesh


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


def boxes_to_mesh(boxes_tensor, labels, scores=None, color_map=None):
    """Convert boxes to trimesh meshes.

    Args:
        boxes_tensor: Nx9 tensor [x, y, z, l, w, h, yaw, vx, vy]
        labels: N tensor of class labels
        scores: N tensor of scores (optional)
        color_map: dict mapping label -> [r, g, b, a]

    Returns:
        list of trimesh.Trimesh
    """
    if color_map is None:
        # Vibrant color map for 7 classes - fully opaque for better visibility
        color_map = {
            0: [255, 50, 50, 255],      # CAR - bright red
            1: [50, 255, 50, 255],      # TRAILER - bright green
            2: [50, 100, 255, 255],     # TRUCK - bright blue
            3: [255, 255, 50, 255],     # VAN - bright yellow
            4: [255, 50, 255, 255],     # PEDESTRIAN - bright magenta
            5: [50, 255, 255, 255],     # BUS - bright cyan
            6: [255, 150, 50, 255],     # BICYCLE - bright orange
        }

    meshes = []
    for i in range(len(boxes_tensor)):
        box = boxes_tensor[i].cpu().numpy() if hasattr(boxes_tensor[i], 'cpu') else boxes_tensor[i]
        label = int(labels[i].cpu().item()) if hasattr(labels[i], 'cpu') else int(labels[i])
        center = box[:3]
        dims = box[3:6]
        rotation = box[6]

        color = color_map.get(label, [200, 200, 200, 255])

        mesh = create_box_mesh(center, dims, rotation, color)
        if mesh is not None:
            meshes.append(mesh)

    return meshes


@HOOKS.register_module()
class GLBVisualizationHook(Hook):
    """Hook to save validation predictions as GLB files.

    Args:
        out_dir (str): Output directory for GLB files
        interval (int): Save every N epochs. Default: 1
        num_samples (int): Number of samples to visualize per epoch. Default: 3
        score_thr (float): Score threshold for predictions. Default: 0.3
    """

    priority = 'NORMAL'

    def __init__(self,
                 out_dir='work_dirs/visualizations',
                 interval=1,
                 num_samples=5,
                 score_thr=0.3,
                 backend_args=None):
        self.out_dir = out_dir
        self.interval = interval
        self.num_samples = num_samples
        self.score_thr = score_thr
        self.backend_args = backend_args
        self._sample_count = 0
        self._current_epoch = -1
        # Note: We can't log here as logger isn't available yet

    def before_val_epoch(self, runner: Runner) -> None:
        """Reset sample counter at start of validation epoch."""
        self._sample_count = 0
        self._current_epoch = runner.epoch

    def before_test_epoch(self, runner: Runner) -> None:
        """Reset sample counter at start of test epoch."""
        self._sample_count = 0
        # For test mode, use epoch 0 or -1 if not in training context
        self._current_epoch = getattr(runner, 'epoch', 0)
        runner.logger.info(
            f'GLBVisualizationHook: Starting test epoch, epoch={self._current_epoch}, '
            f'interval={self.interval}, num_samples={self.num_samples}, '
            f'out_dir={self.out_dir}, score_thr={self.score_thr}'
        )

    def _process_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                      outputs: Sequence, mode: str = 'val') -> None:
        """Common processing for both validation and test iterations.

        Args:
            runner: The runner of the validation/test process.
            batch_idx: The index of the current batch in the loop.
            data_batch: Data from dataloader.
            outputs: A batch of prediction dictionaries.
            mode: 'val' or 'test' to determine directory naming.
        """
        # Only save every N epochs
        # For test mode without training, always visualize (epoch might be -1 or 0)
        # For val mode, check interval based on epoch
        if mode == 'test':
            # In test mode, always visualize (typically run once without training)
            pass
        else:
            # In val mode, check interval
            if (self._current_epoch + 1) % self.interval != 0:
                return

        # Only save first num_samples
        if self._sample_count >= self.num_samples:
            return

        # Debug logging for first iteration
        if batch_idx == 0:
            runner.logger.info(f'GLBVisualizationHook: Processing {mode} iteration, batch_idx={batch_idx}, sample_count={self._sample_count}/{self.num_samples}')

        # Create epoch directory
        if mode == 'test':
            epoch_dir = osp.join(self.out_dir, f'test_epoch_{self._current_epoch}')
        else:
            epoch_dir = osp.join(self.out_dir, f'epoch_{self._current_epoch}')
        os.makedirs(epoch_dir, exist_ok=True)

        # Visualize first sample from batch
        try:
            pred_dict = outputs[0]
            self._visualize_sample(pred_dict, data_batch, epoch_dir, self._sample_count, runner)
            self._sample_count += 1
        except Exception as e:
            runner.logger.warning(f'Failed to visualize sample {self._sample_count}: {e}')
            import traceback
            traceback.print_exc()

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[Det3DDataSample]) -> None:
        """Save visualizations during validation.

        Args:
            runner: The runner of the validation process.
            batch_idx: The index of the current batch in the val loop.
            data_batch: Data from dataloader.
            outputs: A batch of data samples that contain annotations and predictions.
        """
        self._process_iter(runner, batch_idx, data_batch, outputs, mode='val')

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[Det3DDataSample]) -> None:
        """Save visualizations during test.

        Args:
            runner: The runner of the test process.
            batch_idx: The index of the current batch in the test loop.
            data_batch: Data from dataloader.
            outputs: A batch of data samples that contain annotations and predictions.
        """
        self._process_iter(runner, batch_idx, data_batch, outputs, mode='test')

    def _visualize_sample(self, pred_dict: dict, data_batch: dict, save_dir: str,
                         sample_idx: int, runner: Runner):
        """Visualize a single sample and save as GLB.

        Args:
            pred_dict: Prediction dictionary with boxes_3d, scores_3d, labels_3d
            data_batch: Data batch from dataloader
            save_dir: Directory to save GLB file
            sample_idx: Sample index
            runner: Runner for logging
        """
        # Extract predictions from dict
        pred_bboxes = pred_dict.get('boxes_3d')
        pred_labels = pred_dict.get('labels_3d')
        pred_scores = pred_dict.get('scores_3d')

        if pred_bboxes is None or pred_labels is None or pred_scores is None:
            runner.logger.warning(f'Missing predictions in pred_dict: {pred_dict.keys()}')
            return

        # Convert to CPU tensors
        if hasattr(pred_bboxes, 'tensor'):
            pred_bboxes = pred_bboxes.tensor.cpu()
        elif hasattr(pred_bboxes, 'cpu'):
            pred_bboxes = pred_bboxes.cpu()

        if hasattr(pred_labels, 'cpu'):
            pred_labels = pred_labels.cpu()
        if hasattr(pred_scores, 'cpu'):
            pred_scores = pred_scores.cpu()

        # Filter by score threshold
        valid_mask = pred_scores > self.score_thr
        pred_bboxes = pred_bboxes[valid_mask]
        pred_labels = pred_labels[valid_mask]
        pred_scores = pred_scores[valid_mask]

        # Extract GT from data_batch
        gt_bboxes = None
        gt_labels = None
        data_samples = data_batch.get('data_samples', [])
        if len(data_samples) > 0:
            data_sample = data_samples[0]
            if hasattr(data_sample, 'gt_instances_3d'):
                gt_instances = data_sample.gt_instances_3d
                if hasattr(gt_instances, 'bboxes_3d'):
                    gt_bboxes = gt_instances.bboxes_3d.tensor.cpu()
                    gt_labels = gt_instances.labels_3d.cpu()

        # Get point cloud from data_batch
        points = None
        inputs = data_batch.get('inputs', {})
        # Try to get infrastructure points first (usually more complete view)
        if 'infrastructure_points' in inputs:
            points_list = inputs['infrastructure_points']
            if isinstance(points_list, list) and len(points_list) > 0:
                points_tensor = points_list[0]
                if hasattr(points_tensor, 'cpu'):
                    points = points_tensor.cpu().numpy()
                else:
                    points = np.array(points_tensor)
        elif 'vehicle_points' in inputs:
            points_list = inputs['vehicle_points']
            if isinstance(points_list, list) and len(points_list) > 0:
                points_tensor = points_list[0]
                if hasattr(points_tensor, 'cpu'):
                    points = points_tensor.cpu().numpy()
                else:
                    points = np.array(points_tensor)

        runner.logger.info(
            f'Sample {sample_idx}: {len(pred_bboxes)} predictions (>{self.score_thr}), '
            f'{len(gt_bboxes) if gt_bboxes is not None else 0} GT boxes'
        )

        # Create trimesh scene
        scene = trimesh.Scene()
        geometry_count = 0

        # Add point cloud
        if points is not None and len(points) > 0:
            pc_xyz = points[:, :3]
            # Downsample for visualization (max 50k points)
            if len(pc_xyz) > 50000:
                indices = np.random.choice(len(pc_xyz), 50000, replace=False)
                pc_xyz = pc_xyz[indices]

            # Create point cloud (dark gray for better contrast with boxes)
            pc_colors = np.ones((len(pc_xyz), 4), dtype=np.uint8) * [100, 100, 100, 255]
            pc_mesh = trimesh.points.PointCloud(pc_xyz, colors=pc_colors)
            scene.add_geometry(pc_mesh, node_name='point_cloud')
            geometry_count += 1
            runner.logger.info(f'Added point cloud with {len(pc_xyz)} points')

        # Add GT boxes (bright green - fully opaque and distinct)
        gt_added = 0
        if gt_bboxes is not None and len(gt_bboxes) > 0:
            # Bright lime green for GT - very distinctive
            gt_color_map = {i: [0, 255, 0, 255] for i in range(10)}  # All bright green
            gt_meshes = boxes_to_mesh(gt_bboxes, gt_labels, color_map=gt_color_map)
            for i, mesh in enumerate(gt_meshes):
                if mesh is not None:
                    scene.add_geometry(mesh, node_name=f'gt_box_{i}')
                    gt_added += 1
            runner.logger.info(f'Added {gt_added} GT boxes (green)')
            geometry_count += gt_added

        # Add prediction boxes (colored by class - vibrant colors)
        pred_added = 0
        if len(pred_bboxes) > 0:
            pred_meshes = boxes_to_mesh(pred_bboxes, pred_labels, pred_scores)
            for i, mesh in enumerate(pred_meshes):
                if mesh is not None:
                    scene.add_geometry(mesh, node_name=f'pred_box_{i}')
                    pred_added += 1
            runner.logger.info(f'Added {pred_added} prediction boxes (colored by class)')
            geometry_count += pred_added

        runner.logger.info(f'Total geometries in scene: {geometry_count}')

        # Save as GLB
        save_path = osp.join(save_dir, f'sample_{sample_idx}.glb')

        # Validate scene before exporting
        if len(scene.geometry) == 0:
            runner.logger.warning(f'Scene is empty, skipping export for sample {sample_idx}')
            return

        try:
            # Export with error handling
            scene.export(save_path, file_type='glb')
            runner.logger.info(f'Saved visualization to {save_path}')

            # Verify file was created and has size > 0
            import os
            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                runner.logger.info(f'GLB file verified: {os.path.getsize(save_path)} bytes')
            else:
                runner.logger.error(f'GLB file not created or is empty: {save_path}')
        except Exception as e:
            runner.logger.error(f'Failed to export GLB file: {e}')
            import traceback
            traceback.print_exc()


# Debug: Verify hook is registered
print(f"[DEBUG] GLBVisualizationHook module loaded and registered with HOOKS registry")
