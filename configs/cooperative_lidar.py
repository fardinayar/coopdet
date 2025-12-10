# Cooperative LiDAR-only Configuration
# LiDAR-only for both vehicle and infrastructure

_base_ = [
    '_base_/datasets/tumtraf_v2x_cooperative.py',
    '_base_/runtime/mmengine_default.py',
    '_base_/models/cooperative_transfusion.py',
]

# Point cloud configuration (512x512 BEV grid)
voxel_size = [0.293, 0.293, 4]  # Changed from 8 to 4 to match best_paper.pth architecture
point_cloud_range = [-75.0, -75.0, -8.0, 75.0, 75.0, 0.0]

# Model configuration - LiDAR only
model = dict(
    type='CooperativeTransFusionDetector',
    vehicle=dict(
        fusion_model=dict(
            type='BEVFusionHeadless',
            encoders=dict(
                camera=None,  # Disable camera
                lidar=dict(
                    voxelize_reduce=False,
                    voxelize=dict(
                        max_num_points=20,
                        point_cloud_range=point_cloud_range,
                        voxel_size=voxel_size,
                        max_voxels=[30000, 60000],
                        deterministic=False),
                    backbone=dict(
                        type='PointPillarsEncoder',
                        pts_voxel_encoder=dict(
                            type='PillarFeatureNet',
                            in_channels=5,
                            feat_channels=[64, 64],
                            with_distance=False,
                            point_cloud_range=point_cloud_range,
                            voxel_size=voxel_size,
                            norm_cfg=dict(
                                type='BN1d',
                                eps=1.0e-3,
                                momentum=0.01)),
                        pts_middle_encoder=dict(
                            type='PointPillarsScatter',
                            in_channels=64,
                            output_shape=[512, 512])))),
            fuser=None)),  # No fusion needed
    infrastructure=dict(
        fusion_model=dict(
            type='BEVFusionHeadless',
            encoders=dict(
                camera=None,  # Disable camera
                lidar=dict(
                    voxelize_reduce=False,
                    voxelize=dict(
                        max_num_points=20,
                        point_cloud_range=point_cloud_range,
                        voxel_size=voxel_size,
                        max_voxels=[30000, 60000],
                        deterministic=False),
                    backbone=dict(
                        type='PointPillarsEncoder',
                        pts_voxel_encoder=dict(
                            type='PillarFeatureNet',
                            in_channels=5,
                            feat_channels=[64, 64],
                            with_distance=False,
                            point_cloud_range=point_cloud_range,
                            voxel_size=voxel_size,
                            norm_cfg=dict(
                                type='BN1d',
                                eps=1.0e-3,
                                momentum=0.01)),
                        pts_middle_encoder=dict(
                            type='PointPillarsScatter',
                            in_channels=64,
                            output_shape=[512, 512])))),
            fuser=None)),  # No fusion needed
    coop_fuser=dict(
        type='MaxFuser',
        in_channels=[64, 64],  # 64 from each agent's LiDAR encoder
        out_channels=64),  # Output 64 channels
    # Decoder - matching original coopdet3d pointpillars config
    decoder=dict(
        backbone=dict(
            type='SECOND',  # Explicitly set type to override base config
            in_channels=64,  # Matching coop_fuser output
            out_channels=[64, 128, 256],
            layer_nums=[3, 5, 5],
            layer_strides=[2, 2, 2],
            norm_cfg=dict(
                type='BN',
                eps=1.0e-3,
                momentum=0.01),
            conv_cfg=dict(
                type='Conv2d',
                bias=False)),
        neck=dict(
            type='SECONDFPN',  # Explicitly set type to override base config
            in_channels=[64, 128, 256],
            out_channels=[128, 128, 128],
            upsample_strides=[0.5, 1, 2],
            norm_cfg=dict(
                type='BN',
                eps=1.0e-3,
                momentum=0.01),
            upsample_cfg=dict(
                type='deconv',
                bias=False),
            use_conv_for_no_stride=True)),
    # Heads - matching original coopdet3d
    heads=dict(
        object=dict(
            in_channels=384,
            train_cfg=dict(
                grid_size=[512, 512, 1],
                out_size_factor=4,
                point_cloud_range=point_cloud_range,
                voxel_size=voxel_size),
            test_cfg=dict(
                grid_size=[1024, 1024, 1],  # 2x test grid size (original CoopDet3D)
                out_size_factor=8,  # 8 because test grid is 2x train grid
                voxel_size=voxel_size[:2],
                pc_range=point_cloud_range[:2]),
            bbox_coder=dict(
                type='TransFusionBBoxCoder',
                pc_range=point_cloud_range[:2],
                post_center_range=[-85.0, -85.0, -10.0, 85.0, 85.0, 10.0],
                score_threshold=0.01,  # Matching pointpillars_yolov8.py (lowered from 0.1 - model produces scores ~0.004-0.09)
                out_size_factor=4,
                voxel_size=voxel_size[:2],
                code_size=10))))

# Evaluation configuration
val_evaluator = dict(
    type='TUMTrafMetric',
    data_root='data/tumtraf_v2x_cooperative_perception_dataset_processed/',
    ann_file='tumtraf_v2x_nusc_infos_val.pkl',
    metric='bbox',
    modality=dict(use_camera=False, use_lidar=True),
    result_names=['pts_bbox'],
    dataset_type='TUMTrafV2XNuscDataset')
test_evaluator = val_evaluator

# Update dataset to use LiDAR only
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    # Remove camera-related transforms
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=['CAR', 'TRAILER', 'TRUCK', 'VAN', 'PEDESTRIAN', 'BUS', 'BICYCLE']),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
