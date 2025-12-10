# Vehicle-only Fusion Configuration
# Camera + LiDAR fusion for vehicle only (non-cooperative)

_base_ = [
    '_base_/datasets/tumtraf_v2x_cooperative.py',  # Will override to vehicle-only
    '_base_/runtime/mmengine_default.py',
]

# Point cloud configuration
voxel_size = [0.293, 0.293, 4]
point_cloud_range = [-75.0, -75.0, -8.0, 75.0, 75.0, 0.0]

# Model configuration - Single agent (vehicle)
model = dict(
    type='TransFusionDetector',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxelize_cfg=dict(
            max_num_points=20,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=[30000, 60000])),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=5,
        feat_channels=[64, 64],
        with_distance=False,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)),
    pts_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[512, 512]),
    pts_backbone=dict(
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
    pts_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[128, 256, 512],
        out_channels=256,
        num_outs=3,
        start_level=0))

# Override dataset to use vehicle-only data
train_dataloader = dict(
    dataset=dict(
        type='TUMTrafV2XNuscDataset',
        # Filter to vehicle-only samples
        filter_empty_gt=True))

val_dataloader = dict(
    dataset=dict(
        type='TUMTrafV2XNuscDataset',
        filter_empty_gt=False))
