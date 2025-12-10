"""Backbones for cooperative 3D detection."""
from .base_backbone import BaseBackbone
from .yolov8 import YOLOv8CSPDarknet

__all__ = ['BaseBackbone', 'YOLOv8CSPDarknet']

