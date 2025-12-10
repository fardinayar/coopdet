# Default runtime configuration for mmengine
default_scope = 'mmdet3d'

# Environment
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# Logging
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

# Training
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01),  # Increased from 1e-4 to 2e-4
    clip_grad=dict(max_norm=35, norm_type=2),
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.33333333,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=40,  # Extended to match 40 epochs
        end=40,
        by_epoch=True,
        eta_min=0.0002 * 0.01,  # Less aggressive: 2e-4 -> 2e-6 (instead of 1e-4 -> 1e-7)
        convert_to_iter_based=True)
]

# Hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

# Runtime
randomness = dict(seed=0, deterministic=False, diff_rank_seed=False)
