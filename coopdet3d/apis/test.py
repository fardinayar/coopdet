"""Testing APIs."""
import torch

from mmengine.utils import track_iter_progress


def single_gpu_test(model, data_loader):
    """Single GPU test."""
    model.eval()
    results = []
    dataset = data_loader.dataset
    
    # Progress bar - use mmengine track_iter_progress
    data_iter = track_iter_progress(data_loader)
    
    for data in data_iter:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)
    return results
