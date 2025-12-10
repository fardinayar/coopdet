# Infrastructure-only Camera Configuration
# Camera-only for infrastructure (non-cooperative)

_base_ = [
    '_base_/datasets/tumtraf_v2x_cooperative.py',
    '_base_/runtime/mmengine_default.py',
]

# Point cloud configuration (for BEV space)
point_cloud_range = [-75.0, -75.0, -8.0, 75.0, 75.0, 0.0]

# Model configuration - Camera only
model = dict(
    type='BEVFormer',
    use_grid_mask=True,
    img_backbone=dict(
        type='YOLOv8CSPDarknet',
        arch='P5',
        last_stage_out_channels=1024,
        deepen_factor=0.33,
        widen_factor=0.5,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='weights/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1_new.pth')),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[128, 256, 512],
        out_channels=256,
        num_outs=3,
        start_level=0),
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=80,
        image_size=[256, 704],
        feature_size=[32, 88],
        xbound=[-75.0, 75.0, 0.146484375],
        ybound=[-75.0, 75.0, 0.146484375],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2))

# Override dataset to use infrastructure-only, camera-only data
train_dataloader = dict(
    dataset=dict(
        type='TUMTrafV2XNuscDataset',
        modality=dict(use_lidar=False, use_camera=True),
        filter_empty_gt=True))

val_dataloader = dict(
    dataset=dict(
        type='TUMTrafV2XNuscDataset',
        modality=dict(use_lidar=False, use_camera=True),
        filter_empty_gt=False))
