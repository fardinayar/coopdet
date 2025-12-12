"""Loading transforms for cooperative 3D detection."""
import numpy as np
from os import path as osp
from PIL import Image

from mmdet3d.registry import TRANSFORMS as MMDet3D_TRANSFORMS
from mmengine.registry import TRANSFORMS as MMEngine_TRANSFORMS
from mmdet3d.structures.points import BasePoints, get_points_type
from mmdet3d.datasets.transforms import LoadAnnotations3D as MMDet3DLoadAnnotations3D
from mmengine.fileio import get

from .loading_utils import load_augmented_point_cloud, reduce_LiDAR_beams

# Use mmdet3d's TRANSFORMS registry as primary
TRANSFORMS = MMDet3D_TRANSFORMS


@TRANSFORMS.register_module()
class LoadMultiViewImageFromFilesCoop:
    """Load multi channel images from a list of separate channel files for cooperative perception.

    Expects results['vehicle_image_paths'] and results['infrastructure_image_paths'] to be lists of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32. Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
        """
        vehicle_filename = results["vehicle_image_paths"]
        infrastructure_filename = results["infrastructure_image_paths"]

        vehicle_images = []
        infrastructure_images = []

        for vehicle_name in vehicle_filename:
            vehicle_images.append(Image.open(vehicle_name))
        for infrastructure_name in infrastructure_filename:
            infrastructure_images.append(Image.open(infrastructure_name))

        results["vehicle_filename"] = vehicle_filename
        results["infrastructure_filename"] = infrastructure_filename
        results["vehicle_img"] = vehicle_images
        results["infrastructure_img"] = infrastructure_images
        results["vehicle_img_shape"] = vehicle_images[0].size
        results["infrastructure_img_shape"] = infrastructure_images[0].size
        results["vehicle_ori_shape"] = vehicle_images[0].size
        results["infrastructure_ori_shape"] = infrastructure_images[0].size
        results["pad_shape"] = infrastructure_images[0].size
        results["scale_factor"] = 1.0

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@TRANSFORMS.register_module()
class LoadPointsFromFileCoop:
    """Load Points From File for cooperative perception.

    Args:
        coord_type (str): The type of coordinates of points cloud.
        training (bool): Whether in training mode.
        load_dim (int): The dimension of the loaded points. Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        load_augmented (str | None): Type of augmented point cloud. Defaults to None.
        reduce_beams (int | None): Number of beams to reduce to. Defaults to None.
    """

    def __init__(
        self,
        coord_type,
        training,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams
        self.training = training

    def _load_points(self, lidar_path):
        """Private function to load point clouds data."""
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file."""
        vehicle_lidar_path = results["vehicle_lidar_path"]
        infrastructure_lidar_path = results["infrastructure_lidar_path"]
        vehicle_points = self._load_points(vehicle_lidar_path)
        infrastructure_points = self._load_points(infrastructure_lidar_path)
        vehicle_points = vehicle_points.reshape(-1, self.load_dim)
        infrastructure_points = infrastructure_points.reshape(-1, self.load_dim)

        if self.reduce_beams and self.reduce_beams < 32:
            vehicle_points = reduce_LiDAR_beams(vehicle_points, self.reduce_beams)
            infrastructure_points = reduce_LiDAR_beams(infrastructure_points, self.reduce_beams)

        vehicle_points = vehicle_points[:, self.use_dim]
        infrastructure_points = infrastructure_points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            vehicle_floor_height = np.percentile(vehicle_points[:, 2], 0.99)
            infrastructure_floor_height = np.percentile(infrastructure_points[:, 2], 0.99)
            vehicle_height = vehicle_points[:, 2] - vehicle_floor_height
            infrastructure_height = infrastructure_points[:, 2] - infrastructure_floor_height
            vehicle_points = np.concatenate(
                [vehicle_points[:, :3], np.expand_dims(vehicle_height, 1), vehicle_points[:, 3:]], 1
            )
            infrastructure_points = np.concatenate(
                [infrastructure_points[:, :3], np.expand_dims(infrastructure_height, 1), infrastructure_points[:, 3:]], 1
            )
            attribute_dims = dict(vehicle_height=3, infrastructure_height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    vehicle_color=[
                        vehicle_points.shape[1] - 3,
                        vehicle_points.shape[1] - 2,
                        vehicle_points.shape[1] - 1,
                    ],
                    infrastructure_color=[
                        infrastructure_points.shape[1] - 3,
                        infrastructure_points.shape[1] - 2,
                        infrastructure_points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        vehicle_points = points_class(
            vehicle_points, points_dim=vehicle_points.shape[-1], attribute_dims=attribute_dims
        )
        results["vehicle_points"] = vehicle_points
        infrastructure_points = points_class(
            infrastructure_points, points_dim=infrastructure_points.shape[-1], attribute_dims=attribute_dims
        )
        results["infrastructure_points"] = infrastructure_points

        return results


@TRANSFORMS.register_module()
class LoadPointsFromMultiSweepsCoop:
    """Load points from multiple sweeps for cooperative perception.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        pad_empty_sweeps (bool): Whether to repeat keyframe when sweeps is empty.
        remove_close (bool): Whether to remove close points.
        test_mode (bool): If test_model=True used for testing.
        load_augmented (str | None): Type of augmented point cloud.
        reduce_beams (int | None): Number of beams to reduce to.
        training (bool): Whether in training mode.
    """

    def __init__(
        self,
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=False,
        remove_close=False,
        test_mode=False,
        load_augmented=None,
        reduce_beams=None,
        training=False,
    ):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams
        self.training = training

    def _load_points(self, lidar_path):
        """Private function to load point clouds data."""
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin."""
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files."""
        vehicle_points = results["vehicle_points"]
        vehicle_points.tensor[:, 4] = 0
        vehicle_sweep_points_list = [vehicle_points]
        vehicle_ts = results["timestamp"] / 1e6

        infrastructure_points = results["infrastructure_points"]
        infrastructure_points.tensor[:, 4] = 0
        infrastructure_sweep_points_list = [infrastructure_points]
        infrastructure_ts = results["timestamp"] / 1e6

        if self.pad_empty_sweeps and len(results["vehicle_sweeps"]) == 0 and len(results["infrastructure_sweeps"]) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    vehicle_sweep_points_list.append(self._remove_close(vehicle_points))
                    infrastructure_sweep_points_list.append(self._remove_close(infrastructure_points))
                else:
                    vehicle_sweep_points_list.append(vehicle_points)
                    infrastructure_sweep_points_list.append(infrastructure_points)
        else:
            if len(results["vehicle_sweeps"]) <= self.sweeps_num and len(results["infrastructure_sweeps"]) <= self.sweeps_num:
                vehicle_choices = np.arange(len(results["vehicle_sweeps"]))
                infrastructure_choices = np.arange(len(results["infrastructure_sweeps"]))
            elif self.test_mode:
                vehicle_choices = np.arange(self.sweeps_num)
                infrastructure_choices = np.arange(self.sweeps_num)
            else:
                if not self.load_augmented:
                    vehicle_choices = np.random.choice(
                        len(results["vehicle_sweeps"]), self.sweeps_num, replace=False
                    ) if len(results["vehicle_sweeps"]) >= self.sweeps_num else np.arange(len(results["vehicle_sweeps"]))
                    infrastructure_choices = np.random.choice(
                        len(results["infrastructure_sweeps"]), self.sweeps_num, replace=False
                    ) if len(results["infrastructure_sweeps"]) >= self.sweeps_num else np.arange(len(results["infrastructure_sweeps"]))
                else:
                    vehicle_choices = np.random.choice(
                        len(results["vehicle_sweeps"]) - 1, self.sweeps_num, replace=False
                    ) if len(results["vehicle_sweeps"]) > self.sweeps_num else np.arange(len(results["vehicle_sweeps"]))
                    infrastructure_choices = np.random.choice(
                        len(results["infrastructure_sweeps"]) - 1, self.sweeps_num, replace=False
                    ) if len(results["infrastructure_sweeps"]) > self.sweeps_num else np.arange(len(results["infrastructure_sweeps"]))

            for idx in vehicle_choices:
                vehicle_sweep = results["vehicle_sweeps"][idx]
                vehicle_points_sweep = self._load_points(vehicle_sweep["data_path"])
                vehicle_points_sweep = np.copy(vehicle_points_sweep).reshape(-1, self.load_dim)

                if self.reduce_beams and self.reduce_beams < 32:
                    vehicle_points_sweep = reduce_LiDAR_beams(vehicle_points_sweep, self.reduce_beams)

                if self.remove_close:
                    vehicle_points_sweep = self._remove_close(vehicle_points_sweep)
                vehicle_sweep_ts = vehicle_sweep["timestamp"] / 1e6
                vehicle_points_sweep[:, :3] = (
                    vehicle_points_sweep[:, :3] @ vehicle_sweep["sensor2lidar_rotation"].T
                )
                vehicle_points_sweep[:, :3] += vehicle_sweep["sensor2lidar_translation"]
                vehicle_points_sweep[:, 4] = vehicle_ts - vehicle_sweep_ts
                vehicle_points_sweep = vehicle_points.new_point(vehicle_points_sweep)
                vehicle_sweep_points_list.append(vehicle_points_sweep)

            for idy in infrastructure_choices:
                infrastructure_sweep = results["infrastructure_sweeps"][idy]
                infrastructure_points_sweep = self._load_points(infrastructure_sweep["data_path"])
                infrastructure_points_sweep = np.copy(infrastructure_points_sweep).reshape(-1, self.load_dim)

                if self.reduce_beams and self.reduce_beams < 32:
                    infrastructure_points_sweep = reduce_LiDAR_beams(infrastructure_points_sweep, self.reduce_beams)

                if self.remove_close:
                    infrastructure_points_sweep = self._remove_close(infrastructure_points_sweep)
                infrastructure_sweep_ts = infrastructure_sweep["timestamp"] / 1e6
                infrastructure_points_sweep[:, :3] = (
                    infrastructure_points_sweep[:, :3] @ infrastructure_sweep["sensor2lidar_rotation"].T
                )
                infrastructure_points_sweep[:, :3] += infrastructure_sweep["sensor2lidar_translation"]
                infrastructure_points_sweep[:, 4] = infrastructure_ts - infrastructure_sweep_ts
                infrastructure_points_sweep = infrastructure_points.new_point(infrastructure_points_sweep)
                infrastructure_sweep_points_list.append(infrastructure_points_sweep)

        vehicle_points = vehicle_points.cat(vehicle_sweep_points_list)
        vehicle_points = vehicle_points[:, self.use_dim]
        results["vehicle_points"] = vehicle_points
        infrastructure_points = infrastructure_points.cat(infrastructure_sweep_points_list)
        infrastructure_points = infrastructure_points[:, self.use_dim]
        results["infrastructure_points"] = infrastructure_points

        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"


@MMDet3D_TRANSFORMS.register_module()
@MMEngine_TRANSFORMS.register_module()
class LoadPointsFromFileCoopGT:
    """Load Points From File for cooperative perception GT database creation.
    
    This version loads from registered_lidar_path and creates registered_points
    for ground truth database creation.

    Args:
        coord_type (str): The type of coordinates of points cloud.
        load_dim (int): The dimension of the loaded points. Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2].
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        load_augmented (str | None): Type of augmented point cloud. Defaults to None.
        reduce_beams (int | None): Number of beams to reduce to. Defaults to None.
    """

    def __init__(
        self,
        coord_type,
        load_dim=6,
        use_dim=[0, 1, 2],
        shift_height=False,
        use_color=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert (
            max(use_dim) < load_dim
        ), f"Expect all used dimensions < {load_dim}, got {use_dim}"
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.
        
        Args:
            lidar_path (str): Filename of point clouds data.
            
        Returns:
            np.ndarray: An array containing point clouds data.
        """
        # Check file existence (equivalent to mmcv.check_file_exist)
        if not osp.exists(lidar_path):
            raise FileNotFoundError(f"Point cloud file not found: {lidar_path}")
        
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file."""
        registered_lidar_path = results["registered_lidar_path"]
        points = self._load_points(registered_lidar_path)
        points = points.reshape(-1, self.load_dim)

        if self.reduce_beams and self.reduce_beams < 32:
            points = reduce_LiDAR_beams(points, self.reduce_beams)

        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
            )
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        registered_points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims
        )
        results["registered_points"] = registered_points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(coord_type={self.coord_type}, "
        repr_str += f"load_dim={self.load_dim}, "
        repr_str += f"use_dim={self.use_dim})"
        return repr_str


@MMDet3D_TRANSFORMS.register_module()
@MMEngine_TRANSFORMS.register_module()
class LoadPointsFromMultiSweepsCoopGT:
    """Load points from multiple sweeps for cooperative perception GT database creation.
    
    This version works with registered_points for ground truth database creation.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        pad_empty_sweeps (bool): Whether to repeat keyframe when sweeps is empty.
        remove_close (bool): Whether to remove close points.
        test_mode (bool): If test_model=True used for testing.
        load_augmented (str | None): Type of augmented point cloud.
        reduce_beams (int | None): Number of beams to reduce to.
    """

    def __init__(
        self,
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 4],
        pad_empty_sweeps=False,
        remove_close=False,
        test_mode=False,
        load_augmented=None,
        reduce_beams=None,
    ):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.load_augmented = load_augmented
        self.reduce_beams = reduce_beams

    def _load_points(self, lidar_path):
        """Private function to load point clouds data.
        
        Args:
            lidar_path (str): Filename of point clouds data.
            
        Returns:
            np.ndarray: An array containing point clouds data.
        """
        # Check file existence (equivalent to mmcv.check_file_exist)
        if not osp.exists(lidar_path):
            raise FileNotFoundError(f"Point cloud file not found: {lidar_path}")
        
        if self.load_augmented:
            assert self.load_augmented in ["pointpainting", "mvp"]
            virtual = self.load_augmented == "mvp"
            points = load_augmented_point_cloud(
                lidar_path, virtual=virtual, reduce_beams=self.reduce_beams
            )
        elif lidar_path.endswith(".npy"):
            points = np.load(lidar_path)
        else:
            points = np.fromfile(lidar_path, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin."""
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.
        
        Args:
            results (dict): Result dict containing multi-sweep point cloud filenames.
            
        Returns:
            dict: The result dict containing the multi-sweep points data.
        """
        points = results["registered_points"]
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results["timestamp"] / 1e6
        
        if self.pad_empty_sweeps and len(results["registered_sweeps"]) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results["registered_sweeps"]) <= self.sweeps_num:
                choices = np.arange(len(results["registered_sweeps"]))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                # NOTE: seems possible to load frame -11?
                if not self.load_augmented:
                    choices = np.random.choice(
                        len(results["registered_sweeps"]), self.sweeps_num, replace=False
                    )
                else:
                    # don't allow to sample the earliest frame, match with Tianwei's implementation.
                    choices = np.random.choice(
                        len(results["registered_sweeps"]) - 1, self.sweeps_num, replace=False
                    )
            
            for idx in choices:
                sweep = results["registered_sweeps"][idx]
                points_sweep = self._load_points(sweep["data_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)

                # TODO: make it more general
                if self.reduce_beams and self.reduce_beams < 32:
                    points_sweep = reduce_LiDAR_beams(points_sweep, self.reduce_beams)

                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep["timestamp"] / 1e6
                points_sweep[:, :3] = (
                    points_sweep[:, :3] @ sweep["sensor2lidar_rotation"].T
                )
                points_sweep[:, :3] += sweep["sensor2lidar_translation"]
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results["registered_points"] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"


# Re-export LoadAnnotations3D from mmdet3d
# Also register it in mmengine registry for compatibility with mmengine's Compose
LoadAnnotations3D = MMDet3DLoadAnnotations3D
# Register the class in mmengine registry (it's already registered in mmdet3d registry)
try:
    MMEngine_TRANSFORMS.register_module(name='LoadAnnotations3D', module=MMDet3DLoadAnnotations3D, force=False)
except (KeyError, ValueError):
    # Already registered or registration failed, that's okay
    pass

__all__ = [
    'LoadMultiViewImageFromFilesCoop',
    'LoadPointsFromFileCoop',
    'LoadPointsFromMultiSweepsCoop',
    'LoadPointsFromFileCoopGT',
    'LoadPointsFromMultiSweepsCoopGT',
    'LoadAnnotations3D',
]

