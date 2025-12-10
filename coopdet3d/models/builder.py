from mmengine.registry import Registry
from mmengine.config import ConfigDict

# Import mmdet3d registries - deterministic, no fallbacks
from mmdet3d.registry import MODELS, DATASETS

# Use mmdet3d's MODELS registry which includes both mmdet and mmdet3d models
BACKBONES = MODELS  # Use mmdet3d registry which has PillarFeatureNet, etc.
HEADS = MODELS
NECKS = MODELS
LOSSES = MODELS

# Create custom registries for coopdet3d-specific components
# Note: Not using parent=MODELS to avoid scope issues with build_from_cfg
FUSIONMODELS = Registry("fusion_models", scope="coopdet3d")
COOPFUSIONMODELS = Registry("coop_fusion_models", scope="coopdet3d")
VTRANSFORMS = Registry("vtransforms", scope="coopdet3d")
FUSERS = Registry("fusers", scope="coopdet3d")
COOPFUSERS = Registry("coop_fusers", scope="coopdet3d")


def build_backbone(cfg):
    return BACKBONES.build(cfg)


def build_neck(cfg):
    return NECKS.build(cfg)


def build_vtransform(cfg):
    return VTRANSFORMS.build(cfg)


def build_fuser(cfg):
    return FUSERS.build(cfg)


def build_head(cfg):
    return HEADS.build(cfg)


def build_loss(cfg):
    return LOSSES.build(cfg)


def build_fusion_model(cfg, train_cfg=None, test_cfg=None):
    return FUSIONMODELS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg)
    )


def build_model(cfg, train_cfg=None, test_cfg=None):
    return build_fusion_model(cfg, train_cfg=train_cfg, test_cfg=test_cfg)

def build_coop_model(cfg, train_cfg=None, test_cfg=None):
    return build_coop_fusion_model(cfg, train_cfg=train_cfg, test_cfg=test_cfg)


def build_coop_fuser(cfg):
    return COOPFUSERS.build(cfg)


def build_fusion_model_headless(cfg, train_cfg=None, test_cfg=None):
    """Build fusion model without head.
    
    Uses direct class lookup to avoid scope issues with default_scope.
    """
    if isinstance(cfg, ConfigDict):
        cfg = cfg.to_dict()
    else:
        cfg = dict(cfg)  # Make a copy
    model_type = cfg.pop('type')
    cls = FUSIONMODELS.get(model_type)
    if cls is None:
        raise KeyError(f'{model_type} is not registered in FUSIONMODELS')
    return cls(**cfg)


def build_coop_fusion_model(cfg, train_cfg=None, test_cfg=None):
    """Build cooperative fusion model.
    
    Uses direct class lookup to avoid scope issues with default_scope.
    """
    if isinstance(cfg, ConfigDict):
        cfg = cfg.to_dict()
    else:
        cfg = dict(cfg)  # Make a copy
    model_type = cfg.pop('type')
    cls = COOPFUSIONMODELS.get(model_type)
    if cls is None:
        raise KeyError(f'{model_type} is not registered in COOPFUSIONMODELS')
    return cls(**cfg)
