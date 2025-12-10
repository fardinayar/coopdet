# Dataset configuration for TUMTraf V2X Cooperative Perception Dataset
dataset_type = 'TUMTrafV2XNuscDataset'
dataset_root = 'data/tumtraf_v2x_cooperative_perception_dataset_processed/'
gt_paste_stop_epoch = -1
reduce_beams = 32
load_dim = 5
use_dim = 5
load_augmented = None

point_cloud_range = [-75.0, -75.0, -8, 75.0, 75.0, 0]
voxel_size = [0.293, 0.293, 4]  # Matching pointpillars 512x512 grid
image_size = [256, 704]

augment2d = dict(
    resize=[[0.38, 0.55], [0.48, 0.48]],
    rotate=[-5.4, 5.4],
    gridmask=dict(
        prob=0.0,
        fixed_prob=True))

augment3d = dict(
    scale=[0.9, 1.1],
    rotate=[-0.78539816, 0.78539816],
    translate=0.5)

object_classes = [
    'CAR',
    'TRAILER',
    'TRUCK',
    'VAN',
    'PEDESTRIAN',
    'BUS',
    'BICYCLE',
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFilesCoop', to_float32=True),
    dict(
        type='LoadPointsFromFileCoop',
        coord_type='LIDAR',
        training=True,
        load_dim=load_dim,
        use_dim=use_dim,
        reduce_beams=reduce_beams,
        load_augmented=load_augmented),
    dict(
        type='LoadPointsFromMultiSweepsCoop',
        sweeps_num=0,
        load_dim=load_dim,
        use_dim=use_dim,
        reduce_beams=reduce_beams,
        pad_empty_sweeps=True,
        remove_close=True,
        load_augmented=load_augmented,
        training=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='VehiclePointsToInfraCoords'),
    dict(
        type='ObjectPasteCoop',
        stop_epoch=gt_paste_stop_epoch,
        db_sampler=dict(
            dataset_root=dataset_root,
            info_path=dataset_root + 'tumtraf_v2x_nusc_dbinfos_train.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(
                    CAR=5,
                    TRAILER=5,
                    TRUCK=5,
                    VAN=5,
                    PEDESTRIAN=5,
                    BUS=5,
                    BICYCLE=5)),
            classes=object_classes,
            sample_groups=dict(
                CAR=2,
                TRAILER=5,
                TRUCK=3,
                VAN=3,
                PEDESTRIAN=7,
                BUS=5,
                BICYCLE=7),
            points_loader=dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=load_dim,
                use_dim=use_dim))),
    dict(
        type='ImageAug3DCoop',
        final_dim=image_size,
        resize_lim=augment2d['resize'][0],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=augment2d['rotate'],
        rand_flip=True,
        is_train=True),
    dict(
        type='GlobalRotScaleTransCoop',
        resize_lim=augment3d['scale'],
        rot_lim=augment3d['rotate'],
        trans_lim=augment3d['translate'],
        is_train=True),
    dict(
        type='PointsRangeFilterCoop',
        point_cloud_range=point_cloud_range),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=object_classes),
    dict(
        type='ImageNormalizeCoop',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='GridMaskCoop',
        use_h=True,
        use_w=True,
        max_epoch=20,  # Will be overridden by runtime config
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=augment2d['gridmask']['prob'],
        fixed_prob=augment2d['gridmask']['fixed_prob']),
    dict(type='PointShuffleCoop'),
    dict(
        type='DefaultFormatBundle3DCoop',
        classes=object_classes),
    dict(
        type='Collect3DCoop',
        keys=[
            'vehicle_img',
            'infrastructure_img',
            'vehicle_points',
            'infrastructure_points',
            'gt_bboxes_3d',
            'gt_labels_3d'],
        meta_keys=[
            'vehicle_camera_intrinsics',
            'infrastructure_camera_intrinsics',
            'vehicle_lidar2camera',
            'infrastructure_lidar2camera',
            'vehicle_camera2lidar',
            'infrastructure_camera2lidar',
            'vehicle_lidar2image',
            'infrastructure_lidar2image',
            'vehicle_img_aug_matrix',
            'infrastructure_img_aug_matrix',
            'vehicle_lidar_aug_matrix',
            'infrastructure_lidar_aug_matrix',
            'vehicle2infrastructure'])]

val_pipeline = [
    dict(type='LoadMultiViewImageFromFilesCoop', to_float32=True),
    dict(
        type='LoadPointsFromFileCoop',
        coord_type='LIDAR',
        training=False,
        load_dim=load_dim,
        use_dim=use_dim,
        reduce_beams=reduce_beams,
        load_augmented=load_augmented),
    dict(
        type='LoadPointsFromMultiSweepsCoop',
        sweeps_num=0,
        load_dim=load_dim,
        use_dim=use_dim,
        reduce_beams=reduce_beams,
        pad_empty_sweeps=True,
        remove_close=True,
        load_augmented=load_augmented,
        training=False),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='VehiclePointsToInfraCoords'),
    dict(
        type='ImageAug3DCoop',
        final_dim=image_size,
        resize_lim=augment2d['resize'][1],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(
        type='GlobalRotScaleTransCoop',
        resize_lim=[1.0, 1.0],
        rot_lim=[0.0, 0.0],
        trans_lim=0.0,
        is_train=False),
    dict(
        type='PointsRangeFilterCoop',
        point_cloud_range=point_cloud_range),
    dict(
        type='ImageNormalizeCoop',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='DefaultFormatBundle3DCoop',
        classes=object_classes),
    dict(
        type='Collect3DCoop',
        keys=[
            'vehicle_img',
            'infrastructure_img',
            'vehicle_points',
            'infrastructure_points',
            'gt_bboxes_3d',
            'gt_labels_3d'],
        meta_keys=[
            'vehicle_camera_intrinsics',
            'infrastructure_camera_intrinsics',
            'vehicle_lidar2camera',
            'infrastructure_lidar2camera',
            'vehicle_camera2lidar',
            'infrastructure_camera2lidar',
            'vehicle_lidar2image',
            'infrastructure_lidar2image',
            'vehicle_img_aug_matrix',
            'infrastructure_img_aug_matrix',
            'vehicle_lidar_aug_matrix',
            'infrastructure_lidar_aug_matrix',
            'vehicle2infrastructure'])]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesCoop', to_float32=True),
    dict(
        type='LoadPointsFromFileCoop',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=use_dim,
        reduce_beams=reduce_beams,
        load_augmented=load_augmented,
        training=False),
    dict(
        type='LoadPointsFromMultiSweepsCoop',
        sweeps_num=0,
        load_dim=load_dim,
        use_dim=use_dim,
        reduce_beams=reduce_beams,
        pad_empty_sweeps=True,
        remove_close=True,
        load_augmented=load_augmented,
        training=False),
    dict(type='VehiclePointsToInfraCoords'),
    dict(
        type='ImageAug3DCoop',
        final_dim=image_size,
        resize_lim=augment2d['resize'][1],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(
        type='GlobalRotScaleTransCoop',
        resize_lim=[1.0, 1.0],
        rot_lim=[0.0, 0.0],
        trans_lim=0.0,
        is_train=False),
    dict(
        type='PointsRangeFilterCoop',
        point_cloud_range=point_cloud_range),
    dict(
        type='ImageNormalizeCoop',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='DefaultFormatBundle3DCoop',
        classes=object_classes),
    dict(
        type='Collect3DCoop',
        keys=[
            'vehicle_img',
            'infrastructure_img',
            'vehicle_points',
            'infrastructure_points'],
        meta_keys=[
            'vehicle_camera_intrinsics',
            'infrastructure_camera_intrinsics',
            'vehicle_lidar2camera',
            'infrastructure_lidar2camera',
            'vehicle_camera2lidar',
            'infrastructure_camera2lidar',
            'vehicle_lidar2image',
            'infrastructure_lidar2image',
            'vehicle_img_aug_matrix',
            'infrastructure_img_aug_matrix',
            'vehicle_lidar_aug_matrix',
            'infrastructure_lidar_aug_matrix',
            'vehicle2infrastructure'])]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=dataset_root,
            ann_file='tumtraf_v2x_nusc_infos_train.pkl',
            pipeline=train_pipeline,
            object_classes=object_classes,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=True,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file='tumtraf_v2x_nusc_infos_val.pkl',
        pipeline=val_pipeline,
        object_classes=object_classes,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file='tumtraf_v2x_nusc_infos_test.pkl',
        pipeline=test_pipeline,
        object_classes=object_classes,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))

evaluation = dict(interval=1, pipeline=test_pipeline)

