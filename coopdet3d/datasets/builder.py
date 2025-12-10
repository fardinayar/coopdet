"""Dataset builder for cooperative 3D detection."""
import platform

from mmdet3d.registry import DATASETS
from mmdet3d.datasets import CBGSDataset
from mmengine.registry import Registry, build_from_cfg
from mmengine.dataset import ClassBalancedDataset, ConcatDataset, RepeatDataset

if platform.system() != "Windows":
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

OBJECTSAMPLERS = Registry("Object sampler")


def build_dataset(cfg, default_args=None):
    """Build dataset from config.
    
    Args:
        cfg (dict): Config dict for dataset.
        default_args (dict | None): Default arguments for building dataset.
        
    Returns:
        Dataset: The built dataset.
    """
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg["type"] == "ConcatDataset":
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg["datasets"]],
        )
    elif cfg["type"] == "RepeatDataset":
        dataset = RepeatDataset(build_dataset(cfg["dataset"], default_args), cfg["times"])
    elif cfg["type"] == "ClassBalancedDataset":
        dataset = ClassBalancedDataset(
            build_dataset(cfg["dataset"], default_args), cfg["oversample_thr"]
        )
    elif cfg["type"] == "CBGSDataset":
        dataset = CBGSDataset(build_dataset(cfg["dataset"], default_args))
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


__all__ = ['build_dataset', 'OBJECTSAMPLERS', 'DATASETS']
