"""CoopDet3D: Cooperative 3D Object Detection."""

# Import all modules to register components with registries
from . import datasets
from . import evaluation
from . import models
from . import engine

__all__ = ['datasets', 'evaluation', 'models', 'engine']

