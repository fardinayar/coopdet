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
from mmdet3d.models.voxel_encoders import PillarFeatureNet  # noqa: F401
from mmdet3d.models.middle_encoders import PointPillarsScatter  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(description="Test a 3D detection model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--work-dir", help="the dir to save logs and evaluation results")
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

    # Set checkpoint to load
    cfg.load_from = args.checkpoint

    # Build runner and start testing
    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == "__main__":
    main()
