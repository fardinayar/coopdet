"""Hooks for training and validation."""
from .visualization_hook import GLBVisualizationHook

__all__ = ['GLBVisualizationHook']

# Debug: Print to verify import
print(f"[DEBUG] coopdet3d.engine.hooks imported - GLBVisualizationHook: {GLBVisualizationHook}")
