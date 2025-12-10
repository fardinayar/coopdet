# Vehicle-only LiDAR Configuration
# LiDAR-only for vehicle (non-cooperative)

_base_ = [
    '_base_/datasets/tumtraf_v2x_cooperative.py',
    '_base_/runtime/mmengine_default.py',
]

# Point cloud configuration
voxel_size = [0.293, 0.293, 4]
point_cloud_range = [-75.0, -75.0, -8.0, 75.0, 75.0, 0.0]

# Model configuration - LiDAR only
model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxelize_cfg=dict(
            max_num_points=20,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=[30000, 60000])),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=5,
        feat_channels=[64, 64],
        with_distance=False,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)),
    middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[512, 512]),
    backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True))

# Override dataset to use vehicle-only, LiDAR-only data
train_dataloader = dict(
    dataset=dict(
        type='TUMTrafV2XNuscDataset',
        modality=dict(use_lidar=True, use_camera=False),
        filter_empty_gt=True))

val_dataloader = dict(
    dataset=dict(
        type='TUMTrafV2XNuscDataset',
        modality=dict(use_lidar=True, use_camera=False),
        filter_empty_gt=False))
