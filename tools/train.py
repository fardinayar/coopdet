import argparse
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from mmengine import Config
from mmengine.dist import init_dist
from mmengine.runner import Runner
from mmengine.registry import init_default_scope

# Set default scope to mmdet3d
init_default_scope('mmdet3d')

# Import to register components
import coopdet3d.datasets  # noqa: F401
import coopdet3d.evaluation  # noqa: F401
import coopdet3d.engine  # noqa: F401 - Import to register hooks
from mmdet3d.models.voxel_encoders import PillarFeatureNet  # noqa: F401
from mmdet3d.models.middle_encoders import PointPillarsScatter  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Train a 3D detection model")
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
        "in xxx=yyy format will be merged into config file.",
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

    # Ensure default_scope is set
    if hasattr(cfg, 'default_scope') and cfg.default_scope:
        init_default_scope(cfg.default_scope)

    # Override config with command line arguments
    if args.cfg_options is not None:
        cfg.merge_from_dict(dict([opt.split('=') for opt in args.cfg_options]))

    # Set work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # Use config filename as default work_dir
        cfg.work_dir = os.path.join('./work_dirs',
                                    os.path.splitext(os.path.basename(args.config))[0])

    # Set load_from
    if args.load_from is not None:
        cfg.load_from = args.load_from

    # Set resume
    if args.resume:
        cfg.resume = True

    # Enable AMP
    if args.amp:
        optim_wrapper = cfg.optim_wrapper.get('type', 'OptimWrapper')
        if optim_wrapper == 'OptimWrapper':
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # Build runner and start training
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
