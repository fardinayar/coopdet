# Cooperative Fusion Configuration
# Camera + LiDAR fusion for both vehicle and infrastructure

_base_ = [
    '_base_/datasets/divp.py',
    '_base_/runtime/mmengine_default.py',
    '_base_/models/cooperative_transfusion.py',
]

# Point cloud configuration (512x512 BEV grid)
voxel_size = [0.293, 0.293, 4]  # Original CoopDet3D uses 4, not 8
point_cloud_range = [-75.0, -75.0, -8.0, 75.0, 75.0, 0.0]

# Model configuration - Camera + LiDAR fusion
model = dict(
    type='CooperativeTransFusionDetector',
    vehicle=dict(
        fusion_model=dict(
            type='BEVFusionHeadless',
            encoders=dict(
                camera=dict(
                    backbone=dict(
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
                    neck=dict(
                        type='GeneralizedLSSFPN',
                        in_channels=[128, 256, 512],
                        out_channels=256,
                        num_outs=3,
                        start_level=0),
                    vtransform=dict(
                        type='CoopDepthLSSTransform',
                        in_channels=256,
                        out_channels=80,
                        image_size=[256, 704],
                        feature_size=[32, 88],
                        xbound=[-75.0, 75.0, 0.146484375],
                        ybound=[-75.0, 75.0, 0.146484375],
                        zbound=[-10.0, 10.0, 20.0],
                        dbound=[1.0, 60.0, 0.5],
                        downsample=2,
                        vehicle=True)),
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
            fuser=dict(
                type='ConvFuser',
                in_channels=[80, 64],
                out_channels=64))),  # Output 64 channels
    infrastructure=dict(
        fusion_model=dict(
            type='BEVFusionHeadless',
            encoders=dict(
                camera=dict(
                    backbone=dict(
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
                    neck=dict(
                        type='GeneralizedLSSFPN',
                        in_channels=[128, 256, 512],
                        out_channels=256,
                        num_outs=3,
                        start_level=0),
                    vtransform=dict(
                        type='CoopDepthLSSTransform',
                        in_channels=256,
                        out_channels=80,
                        image_size=[256, 704],
                        feature_size=[32, 88],
                        xbound=[-75.0, 75.0, 0.146484375],
                        ybound=[-75.0, 75.0, 0.146484375],
                        zbound=[-10.0, 10.0, 20.0],
                        dbound=[1.0, 60.0, 0.5],
                        downsample=2,
                        vehicle=False)),
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
            fuser=dict(
                type='ConvFuser',
                in_channels=[80, 64],
                out_channels=64))),  # Output 64 channels
    coop_fuser=dict(
        type='MaxFuser',
        in_channels=[64, 64],  # 64 from each agent fuser
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
                grid_size=[512, 512, 1],  # Match train grid size
                out_size_factor=4,  # Match train out_size_factor
                voxel_size=voxel_size[:2],
                pc_range=point_cloud_range[:2]),
            bbox_coder=dict(
                type='TransFusionBBoxCoder',
                pc_range=point_cloud_range[:2],
                post_center_range=[-85.0, -85.0, -10.0, 85.0, 85.0, 10.0],
                score_threshold=0.05,  # Balanced threshold between 0.01 and 0.1
                out_size_factor=4,
                voxel_size=voxel_size[:2],
                code_size=10))))

# Evaluation configuration
val_evaluator = dict(
    type='DIVPMetric',
    data_root='data/divp_dataset_processed/',
    ann_file='divp_v2x_nusc_infos_val.pkl',
    metric='bbox',
    modality=dict(use_camera=True, use_lidar=True),
    result_names=['pts_bbox'],
    dataset_type='DIVPNuscDataset')
test_evaluator = val_evaluator

# Custom hooks for visualization
custom_hooks = [
    dict(
        type='GLBVisualizationHook',
        out_dir='work_dirs/visualizations',
        interval=1,  # Save every epoch
        num_samples=5,  # Visualize 5 samples per epoch
        score_thr=0.3)  # Only show predictions with score > 0.3
]
