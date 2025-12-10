"""Models for cooperative 3D detection."""
from .builder import (
    build_backbone,
    build_neck,
    build_vtransform,
    build_fuser,
    build_head,
    build_loss,
    build_fusion_model,
    build_model,
    build_coop_model,
    build_coop_fuser,
    build_fusion_model_headless,
    build_coop_fusion_model,
    MODELS,
    BACKBONES,
    HEADS,
    NECKS,
    LOSSES,
    FUSIONMODELS,
    COOPFUSIONMODELS,
    VTRANSFORMS,
    FUSERS,
    COOPFUSERS,
)

# Import submodules to register components
from . import backbones
from . import coop_fusion_models
from . import fusion_models
from . import coop_fusers
from . import fusers
from . import heads
from . import necks
from . import task_modules
from . import vtransforms
from . import voxel_encoders

__all__ = [
    'build_backbone',
    'build_neck',
    'build_vtransform',
    'build_fuser',
    'build_head',
    'build_loss',
    'build_fusion_model',
    'build_model',
    'build_coop_model',
    'build_coop_fuser',
    'build_fusion_model_headless',
    'build_coop_fusion_model',
    'MODELS',
    'BACKBONES',
    'HEADS',
    'NECKS',
    'LOSSES',
    'FUSIONMODELS',
    'COOPFUSIONMODELS',
    'VTRANSFORMS',
    'FUSERS',
    'COOPFUSERS',
]

