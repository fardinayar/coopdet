"""Training APIs for cooperative 3D detection models.

This module provides training functions for cooperative 3D detection models.
Uses the new mmengine Runner API.
"""
import torch
from torch.utils.data import DataLoader

# Import from mmengine (new API)
from mmengine.runner import Runner
from mmengine.registry import RUNNERS
from mmengine.dataset import DefaultSampler

# Import from coopdet3d
from coopdet3d.datasets import build_dataset
from coopdet3d.utils import get_root_logger
# Import hooks to ensure they are registered
# This must be imported before Runner processes custom_hooks from config
import coopdet3d.engine.hooks  # noqa: F401


def _convert_old_config_to_new(cfg, validate=False):
    """Convert old config format to new mmengine format.
    
    Converts:
    - cfg.optimizer -> cfg.optim_wrapper
    - cfg.runner -> cfg.train_cfg
    """
    # Convert optimizer to optim_wrapper
    if (hasattr(cfg, 'optimizer') and cfg.optimizer is not None and 
        (not hasattr(cfg, 'optim_wrapper') or cfg.optim_wrapper is None)):
        cfg.optim_wrapper = dict(
            type='OptimWrapper',
            optimizer=cfg.optimizer
        )
    
    # Convert runner to train_cfg
    if (hasattr(cfg, 'runner') and cfg.runner is not None and
        (not hasattr(cfg, 'train_cfg') or cfg.train_cfg is None)):
        runner_cfg = cfg.runner
        # Try to get max_epochs from runner, then from top-level config, then default
        max_epochs = runner_cfg.get('max_epochs', None)
        if max_epochs is None:
            max_epochs = getattr(cfg, 'max_epochs', 20)
        if max_epochs is None or max_epochs == 0:
            max_epochs = 20  # Default to 20 if not set or 0
        
        val_interval = runner_cfg.get('val_interval', 1) if validate else None
        
        train_cfg_dict = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs)
        if val_interval is not None:
            train_cfg_dict['val_interval'] = val_interval
        cfg.train_cfg = train_cfg_dict
    elif (not hasattr(cfg, 'train_cfg') or cfg.train_cfg is None):
        # If no runner config, create train_cfg from top-level max_epochs
        max_epochs = getattr(cfg, 'max_epochs', 20)
        if max_epochs is None or max_epochs == 0:
            max_epochs = 20
        
        train_cfg_dict = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs)
        if validate:
            train_cfg_dict['val_interval'] = 1
        cfg.train_cfg = train_cfg_dict


def _build_dataloader_from_dataset(dataset, cfg):
    """Build dataloader from dataset using old config format."""
    samples_per_gpu = getattr(cfg.data, 'samples_per_gpu', 1) if hasattr(cfg, 'data') else 1
    workers_per_gpu = getattr(cfg.data, 'workers_per_gpu', 1) if hasattr(cfg, 'data') else 1
    
    # Debug logging
    logger = get_root_logger()
    logger.info(f"Building training dataloader: samples_per_gpu={samples_per_gpu}, workers_per_gpu={workers_per_gpu}")
    if hasattr(cfg, 'data'):
        logger.info(f"cfg.data.samples_per_gpu = {getattr(cfg.data, 'samples_per_gpu', 'NOT SET')}")
        logger.info(f"cfg.data.workers_per_gpu = {getattr(cfg.data, 'workers_per_gpu', 'NOT SET')}")
    
    # Use Runner.build_dataloader to build dataloader
    dataloader_cfg = dict(
        dataset=dataset,
        sampler=dict[str, str | bool](type='DefaultSampler', shuffle=True),
        batch_size=samples_per_gpu,
        num_workers=workers_per_gpu,
    )
    logger.info(f"dataloader_cfg batch_size = {dataloader_cfg['batch_size']}")
    dataloader = Runner.build_dataloader(dataloader_cfg)
    logger.info(f"Built dataloader with batch_size = {dataloader.batch_size}")
    return dataloader


def train_model(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
):
    """Train a standard 3D detection model using the new mmengine API.

    Args:
        model (nn.Module): The model to train.
        dataset: Training dataset.
        cfg: Config object.
        distributed (bool): Whether to use distributed training.
        validate (bool): Whether to run validation.
        timestamp (str): Timestamp for logging.
    """
    logger = get_root_logger()

    # Convert old config format to new format
    _convert_old_config_to_new(cfg, validate=validate)
    
    # Log conversion results for debugging
    if hasattr(cfg, 'train_cfg'):
        logger.info(f"train_cfg: {cfg.train_cfg}")
    if hasattr(cfg, 'optim_wrapper'):
        logger.info(f"optim_wrapper type: {cfg.optim_wrapper.get('type', 'N/A')}")
    
    # Build dataloader from dataset
    train_dataloader = None
    if dataset is not None:
        train_dataset = dataset[0] if isinstance(dataset, (list, tuple)) and len(dataset) > 0 else dataset
        train_dataloader = _build_dataloader_from_dataset(train_dataset, cfg)
    
    # Build validation dataloader if needed
    val_dataloader = None
    val_cfg = None
    val_evaluator = None
    if validate and hasattr(cfg, 'data') and hasattr(cfg.data, 'val'):
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        # samples_per_gpu should not be in dataset config, use top-level or default to training batch size
        val_samples_per_gpu = getattr(cfg.data.val, 'samples_per_gpu',
            getattr(cfg.data, 'samples_per_gpu', 1))
        val_workers_per_gpu = getattr(cfg.data, 'workers_per_gpu', 1)
        val_dataloader_cfg = dict(
            dataset=val_dataset,
            sampler=dict(type='DefaultSampler', shuffle=False),
            batch_size=val_samples_per_gpu,
            num_workers=val_workers_per_gpu,
        )
        val_dataloader = Runner.build_dataloader(val_dataloader_cfg)
        val_cfg = dict(type='ValLoop')
        # val_evaluator can be passed as dict/list - Runner will build it
        # or as None/empty list if not configured
        if hasattr(cfg, 'val_evaluator') and cfg.val_evaluator is not None:
            val_evaluator = cfg.val_evaluator
        else:
            # If no evaluator config, use empty list (acceptable by Runner)
            val_evaluator = []
    
    # Set work_dir if not set
    if not hasattr(cfg, 'work_dir') or cfg.work_dir is None:
        cfg.work_dir = cfg.get('run_dir', './work_dirs')
    
    # Build runner directly with pre-built components
    runner = Runner(
            model=model,
        work_dir=cfg.work_dir,
        train_dataloader=train_dataloader,
        train_cfg=cfg.train_cfg if hasattr(cfg, 'train_cfg') else None,
        optim_wrapper=cfg.optim_wrapper if hasattr(cfg, 'optim_wrapper') else None,
        val_dataloader=val_dataloader,
        val_cfg=val_cfg,
        val_evaluator=val_evaluator,
        cfg=cfg,
    )
    
    # Store timestamp if needed
    if timestamp is not None:
        runner.infra_timestamp = timestamp

    # Start training
    logger.info(f"Starting training with max_epochs={cfg.train_cfg.get('max_epochs', 'unknown')}")
    runner.train()


def train_model_coop(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
    freeze=False,
    load_from=None,
):
    """Train a cooperative 3D detection model using the new mmengine API.

    Args:
        model (nn.Module): The cooperative model to train.
        dataset: Training dataset.
        cfg: Config object.
        distributed (bool): Whether to use distributed training.
        validate (bool): Whether to run validation.
        timestamp (str): Timestamp for logging.
        freeze (bool): Whether to freeze backbone parameters.
        load_from (str, optional): Path to checkpoint file to load from.
    """
    logger = get_root_logger()

    # Handle parameter freezing
    if freeze:
        logger.info("Freezing backbone parameters")
        for name, param in model.named_parameters():
            if "vehicle" not in name and "infrastructure" not in name:
                param.requires_grad = False
    
    # Convert old config format to new format
    _convert_old_config_to_new(cfg, validate=validate)
    
    # Check if we should skip training (for debugging - run evaluation only)
    # If max_epochs is 0, automatically enable skip_training to run evaluation only
    skip_training = cfg.get('skip_training', False)
    if hasattr(cfg, 'train_cfg'):
        logger.info(f"train_cfg: {cfg.train_cfg}")
        max_epochs = cfg.train_cfg.get('max_epochs', 0)
        if max_epochs == 0:
            if not skip_training:
                # If max_epochs is 0 and skip_training not explicitly set, enable it
                skip_training = True
                logger.info("max_epochs is 0 - automatically enabling skip_training to run evaluation only")
            else:
                logger.info("max_epochs is 0 and skip_training=True - skipping training, running evaluation only")
        elif skip_training:
            logger.info("skip_training=True - skipping training, running evaluation only")
    if hasattr(cfg, 'optim_wrapper'):
        logger.info(f"optim_wrapper type: {cfg.optim_wrapper.get('type', 'N/A')}")
    
    # Build dataloader from dataset (only if not skipping training)
    train_dataloader = None
    if not skip_training and dataset is not None:
        train_dataset = dataset[0] if isinstance(dataset, (list, tuple)) and len(dataset) > 0 else dataset
        train_dataloader = _build_dataloader_from_dataset(train_dataset, cfg)
    
    # Build validation dataloader - always build if validate=True or skip_training=True
    val_dataloader = None
    val_cfg = None
    val_evaluator = None
    if (validate or skip_training) and hasattr(cfg, 'data') and hasattr(cfg.data, 'val'):
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        # samples_per_gpu should not be in dataset config, use top-level or default to training batch size
        val_samples_per_gpu = getattr(cfg.data.val, 'samples_per_gpu', 
            getattr(cfg.data, 'samples_per_gpu', 1))
        val_workers_per_gpu = getattr(cfg.data, 'workers_per_gpu', 1)
        val_dataloader_cfg = dict(
            dataset=val_dataset,
            sampler=dict(type='DefaultSampler', shuffle=False),
            batch_size=val_samples_per_gpu,
            num_workers=val_workers_per_gpu,
        )
        val_dataloader = Runner.build_dataloader(val_dataloader_cfg)
        val_cfg = dict(type='ValLoop')
        # val_evaluator can be passed as dict/list - Runner will build it
        # or as None/empty list if not configured
        if hasattr(cfg, 'val_evaluator') and cfg.val_evaluator is not None:
            val_evaluator = cfg.val_evaluator
        else:
            # If no evaluator config, use empty list (acceptable by Runner)
            val_evaluator = []
    
    # Set work_dir if not set
    if not hasattr(cfg, 'work_dir') or cfg.work_dir is None:
        cfg.work_dir = cfg.get('run_dir', './work_dirs')

    # Build custom hooks if present in config
    # MMEngine Runner doesn't automatically build custom_hooks when constructed manually,
    # so we need to build them explicitly
    custom_hooks_list = None
    if hasattr(cfg, 'custom_hooks') and cfg.custom_hooks is not None:
        from mmengine.registry import HOOKS
        custom_hooks_list = []
        for hook_cfg in cfg.custom_hooks:
            hook = HOOKS.build(hook_cfg)
            custom_hooks_list.append(hook)
            logger.info(f"Built custom hook: {hook.__class__.__name__}")

    # Build runner directly with pre-built components
    runner_kwargs = dict(
        model=model,
        work_dir=cfg.work_dir,
        train_dataloader=train_dataloader,
        train_cfg=cfg.train_cfg if (hasattr(cfg, 'train_cfg') and not skip_training) else None,
        optim_wrapper=cfg.optim_wrapper if (hasattr(cfg, 'optim_wrapper') and not skip_training) else None,
        val_dataloader=val_dataloader,
        val_cfg=val_cfg,
        val_evaluator=val_evaluator,
        custom_hooks=custom_hooks_list,
        cfg=cfg,
    )
    # Add load_from if provided
    if load_from is not None:
        # Manually load checkpoint and filter camera keys to avoid warnings
        # Camera encoder weights are loaded via init_cfg from YOLOv8 checkpoint
        import torch
        
        logger.info(f"Loading checkpoint from: {load_from}")
        checkpoint = torch.load(load_from, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            
            # Filter out camera encoder keys (loaded from YOLOv8 via init_cfg)
            camera_patterns = [
                'nodes.vehicle.fusion_model.encoders.camera.backbone',
                'nodes.infrastructure.fusion_model.encoders.camera.backbone',
                'nodes.vehicle.fusion_model.encoders.camera.neck',
                'nodes.infrastructure.fusion_model.encoders.camera.neck',
                'nodes.vehicle.fusion_model.encoders.camera.vtransform',
                'nodes.infrastructure.fusion_model.encoders.camera.vtransform',
            ]
            
            filtered_state_dict = {}
            camera_keys_removed = []
            
            for key, value in state_dict.items():
                is_camera_key = any(pattern in key for pattern in camera_patterns)
                if is_camera_key:
                    camera_keys_removed.append(key)
                else:
                    filtered_state_dict[key] = value
            
            if camera_keys_removed:
                logger.info(f"Filtered out {len(camera_keys_removed)} camera encoder keys from checkpoint "
                           f"(these are loaded from YOLOv8 checkpoint via init_cfg)")
            
            # Update checkpoint with filtered state_dict
            checkpoint['state_dict'] = filtered_state_dict
        
        # Load checkpoint directly into model to avoid mmengine's warning
        # Use load_state_dict directly since we already have the checkpoint dict
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("Checkpoint loaded successfully (camera encoder keys filtered)")
        
        # Verify weights were loaded correctly
        if missing_keys:
            logger.warning(f"Missing keys in model (not loaded): {len(missing_keys)}")
            if len(missing_keys) <= 10:
                logger.debug(f"Missing keys: {missing_keys}")
            else:
                logger.debug(f"Missing keys (first 10): {missing_keys[:10]}...")
        else:
            logger.info("All checkpoint keys found in model")
        
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint (not in model): {len(unexpected_keys)}")
            if len(unexpected_keys) <= 10:
                logger.debug(f"Unexpected keys: {unexpected_keys}")
            else:
                logger.debug(f"Unexpected keys (first 10): {unexpected_keys[:10]}...")
        
        # Verify loaded weights match checkpoint (sample check)
        model_state_dict = dict(model.named_parameters())
        model_buffers = dict(model.named_buffers())
        model_all = {**model_state_dict, **model_buffers}
        
        matched_count = 0
        shape_mismatch_count = 0
        value_mismatch_count = 0
        
        for key in list(checkpoint['state_dict'].keys())[:100]:  # Sample first 100 keys
            if key in model_all:
                ckpt_tensor = checkpoint['state_dict'][key]
                model_tensor = model_all[key]
                # Move checkpoint tensor to same device as model tensor
                if ckpt_tensor.device != model_tensor.device:
                    ckpt_tensor = ckpt_tensor.to(model_tensor.device)
                if ckpt_tensor.shape == model_tensor.shape:
                    if torch.allclose(ckpt_tensor, model_tensor, rtol=1e-5, atol=1e-8):
                        matched_count += 1
                    else:
                        value_mismatch_count += 1
                else:
                    shape_mismatch_count += 1
        
        total_sampled = min(100, len(checkpoint['state_dict']))
        logger.info(f"Weight verification (sampled {total_sampled} keys): "
                   f"{matched_count} matched, {shape_mismatch_count} shape mismatch, "
                   f"{value_mismatch_count} value mismatch")
        
        if shape_mismatch_count > 0 or value_mismatch_count > 0:
            logger.warning("Some weights may not have loaded correctly. "
                          "Run tools/verify_weights_loaded.py for detailed verification.")
        
        # Don't pass load_from to Runner since we've already loaded it
        # runner_kwargs['load_from'] = load_from  # Commented out - already loaded
    elif skip_training:
        # If skipping training, we need a checkpoint to load
        logger.warning("skip_training=True but no load_from specified. Model will use random weights.")
    
    runner = Runner(**runner_kwargs)
    
    # Store timestamp if needed
    if timestamp is not None:
        runner.infra_timestamp = timestamp
    
    # Start training or validation
    if skip_training:
        logger.info("Skipping training - running evaluation only")
        if val_dataloader is None:
            raise ValueError("skip_training=True but no validation dataloader configured. "
                           "Please ensure cfg.data.val is set and validate=True.")
        runner.val()
    else:
        logger.info(f"Starting training with max_epochs={cfg.train_cfg.get('max_epochs', 'unknown')}")
        runner.train()
