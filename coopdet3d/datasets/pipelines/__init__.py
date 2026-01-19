"""Pipeline transforms for cooperative 3D detection."""
from .loading import (
    LoadMultiViewImageFromFilesCoop,
    LoadPointsFromFileCoop,
    LoadPointsFromMultiSweepsCoop,
    LoadPointsFromFileCoopGT,
    LoadPointsFromMultiSweepsCoopGT,
    LoadAnnotations3D,
)
from .transforms_3d import (
    ImageAug3DCoop,
    GlobalRotScaleTransCoop,
    VehiclePointsToInfraCoords,
    GridMaskCoop,
    ObjectPasteCoop,
    PointShuffleCoop,
    PointsRangeFilterCoop,
    ImageNormalizeCoop,
)
from .formating import (
    DefaultFormatBundle3DCoop,
    Collect3DCoop,
)
from .dbsampler import DataBaseSampler

# Import standard transforms from mmdet3d (already registered)
from mmdet3d.datasets.transforms import (
    ObjectRangeFilter,
    ObjectNameFilter,
)

# Register mmdet3d standard transforms in mmengine registry for compatibility
from mmengine.registry import TRANSFORMS as MMEngine_TRANSFORMS

_standard_transforms = [
    ('ObjectRangeFilter', ObjectRangeFilter),
    ('ObjectNameFilter', ObjectNameFilter),
]

for name, transform_class in _standard_transforms:
    try:
        MMEngine_TRANSFORMS.register_module(name=name, module=transform_class, force=False)
    except (KeyError, ValueError):
        # Already registered or registration failed, that's okay
        pass

__all__ = [
    # Loading
    'LoadMultiViewImageFromFilesCoop',
    'LoadPointsFromFileCoop',
    'LoadPointsFromMultiSweepsCoop',
    'LoadPointsFromFileCoopGT',
    'LoadPointsFromMultiSweepsCoopGT',
    'LoadAnnotations3D',
    # Transforms (custom for coop)
    'ImageAug3DCoop',
    'GlobalRotScaleTransCoop',
    'VehiclePointsToInfraCoords',
    'GridMaskCoop',
    'ObjectPasteCoop',
    'PointShuffleCoop',
    'PointsRangeFilterCoop',
    'ImageNormalizeCoop',
    # Transforms (from mmdet3d)
    'ObjectRangeFilter',
    'ObjectNameFilter',
    # Formatting
    'DefaultFormatBundle3DCoop',
    'Collect3DCoop',
    # Samplers
    'DataBaseSampler',
]

