import argparse
import copy
import os
import random
import sys
import time

# Add project root to path so coopdet3d can be imported
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
from mmengine import Config
from mmengine.dist import init_dist, get_local_rank
from mmengine.runner import set_random_seed
from mmengine.registry import DefaultScope, init_default_scope

# Set default scope to mmdet3d BEFORE importing coopdet3d
# This ensures all registrations go to the correct scope
init_default_scope('mmdet3d')

# Import coopdet3d to ensure all components are registered
import coopdet3d.datasets  # noqa: F401
import coopdet3d.evaluation  # noqa: F401 - Import to register metrics
# Import hooks to ensure they are registered
# This must be imported before Runner processes custom_hooks from config
import coopdet3d.engine.hooks  # noqa: F401

from coopdet3d.apis import train_model_coop
from coopdet3d.datasets import build_dataset
from coopdet3d.models import build_coop_model
from coopdet3d.utils import get_root_logger

# Import mmdet3d models to ensure they're registered
from mmdet3d.models.voxel_encoders import PillarFeatureNet  # noqa: F401
from mmdet3d.models.middle_encoders import PointPillarsScatter  # noqa: F401
from torch.nn import SyncBatchNorm
convert_sync_batchnorm = SyncBatchNorm.convert_sync_batchnorm


def parse_args():
    parser = argparse.ArgumentParser(description="Train a cooperative detection model")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--load-from",
        help="the checkpoint file to load from",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from the latest checkpoint in the work_dir automatically",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="enable automatic-mixed-precision training",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        "be overwritten is a list, it should be like key=\"[a,b]\" or key=a,b "
        "It also allows nested list/tuple values, e.g. key=\"[(a,b),(c,d)]\" "
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # Initialize distributed training
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **{"backend": "nccl"})

    # Load config
    cfg = Config.fromfile(args.config)
    
    # Ensure default_scope is set after loading config
    # This is important for registry lookups (e.g., metrics)
    if hasattr(cfg, 'default_scope') and cfg.default_scope:
        init_default_scope(cfg.default_scope)
    
    if args.cfg_options is not None:
        # Parse cfg-options from command line (format: key=value)
        cfg_options = {}
        for opt in args.cfg_options:
            if "=" in opt:
                key, value = opt.split("=", 1)
                # Try to evaluate as Python literal, otherwise keep as string
                try:
                    import ast
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
                cfg_options[key] = value
        cfg.merge_from_dict(cfg_options)

    # Set work directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # Use config filename as default work_dir if cfg.work_dir is None
        work_dir = os.path.join("./work_dirs", os.path.splitext(os.path.basename(args.config))[0])
        cfg.work_dir = work_dir
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # For backward compatibility with train_model_coop which uses cfg.run_dir
    cfg.run_dir = cfg.work_dir

    # Set cudnn benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # Set device
    if distributed:
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
    else:
        # For non-distributed training, ensure CUDA is available and set device
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    # Dump config
    cfg.dump(os.path.join(cfg.work_dir, "config.yaml"))

    # Init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    # Log some basic info
    logger.info(f"Config:\n{cfg.pretty_text}")

    # Set random seeds
    seed = cfg.get("seed", None)
    deterministic = cfg.get("deterministic", False)
    if seed is not None:
        logger.info(
            f"Set random seed to {seed}, "
            f"deterministic mode: {deterministic}"
        )
        set_random_seed(seed, deterministic=deterministic)

    # Build dataset  
    # Support both old format (cfg.data.train) and new format (cfg.train_dataloader)
    if hasattr(cfg, 'data') and cfg.data is not None:
        # Old format: cfg.data.train
        datasets = [build_dataset(cfg.data.train)]
    elif hasattr(cfg, 'train_dataloader') and cfg.train_dataloader is not None:
        # New format: cfg.train_dataloader.dataset
        datasets = [build_dataset(cfg.train_dataloader.dataset)]
    else:
        raise ValueError(
            "Config must have either 'data.train' (old format) or "
            "'train_dataloader.dataset' (new format) defined"
        )

    # Build model
    model = build_coop_model(cfg.model)
    model.init_weights()
    
    # Convert sync batchnorm
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    # Move model to GPU if available
    if torch.cuda.is_available():
        if distributed:
            device = torch.device(f'cuda:{get_local_rank()}')
        else:
            device = torch.device('cuda:0')
        model = model.to(device)
        logger.info(f"Model moved to device: {device}")
    else:
        logger.warning("CUDA is not available, model will run on CPU")

    logger.info(f"Model:\n{model}")
    
    # Train model
    train_model_coop(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=True,
        timestamp=timestamp,
        freeze=cfg.get("freeze", False),
        load_from=getattr(args, 'load_from', None)
    )


if __name__ == "__main__":
    main()
