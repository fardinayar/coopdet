"""Pipeline transforms for cooperative 3D detection."""
from .loading import (
    LoadMultiViewImageFromFilesCoop,
    LoadPointsFromFileCoop,
    LoadPointsFromMultiSweepsCoop,
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

__all__ = [
    # Loading
    'LoadMultiViewImageFromFilesCoop',
    'LoadPointsFromFileCoop',
    'LoadPointsFromMultiSweepsCoop',
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

