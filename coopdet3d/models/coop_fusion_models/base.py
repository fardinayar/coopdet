from abc import ABCMeta
from typing import Union, List, Dict, Any

import torch
from mmengine.model import BaseModel

__all__ = ["Base3DCoopFusionModel"]


class Base3DCoopFusionModel(BaseModel, metaclass=ABCMeta):
    """Base class for cooperative fusion models.
    
    This class follows mmdet3d's Base3DDetector interface pattern with
    forward() method supporting 'loss', 'predict', and 'tensor' modes.
    """

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.fp16_enabled = False

    def forward(
        self,
        inputs: Union[dict, List[dict]],
        data_samples=None,
        mode: str = 'tensor',
        **kwargs
    ):
        """The unified entry for a forward process in both training and test.
        
        This method follows mmdet3d's Base3DDetector interface pattern.

        Args:
            inputs (dict | list[dict]): Input data dict containing model inputs.
            data_samples (list, optional): Data samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.
                - 'loss': Forward and return a dict of losses.
                - 'predict': Forward and return predictions.
                - 'tensor': Forward and return tensor or tuple of tensor.

        Returns:
            The return type depends on ``mode``.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                             'Only supports loss, predict and tensor mode')

    def _forward(self, inputs, data_samples=None, **kwargs):
        """Forward the whole network and return tensor or tuple of tensor.
        
        This is the base implementation. Subclasses should override this
        if they need custom forward behavior.
        """
        raise NotImplementedError('Subclasses must implement _forward()')

    def loss(self, inputs, data_samples=None, **kwargs):
        """Forward and return a dict of losses.
        
        This is the base implementation. Subclasses should override this.
        """
        raise NotImplementedError('Subclasses must implement loss()')

    def predict(self, inputs, data_samples=None, **kwargs):
        """Forward and return predictions.
        
        This is the base implementation. Subclasses should override this.
        """
        raise NotImplementedError('Subclasses must implement predict()')

    def _preprocess_data(self, data):
        """Preprocess data from dataloader format to model input format.

        This method converts the dataloader output (with 'inputs' and 'data_samples')
        into the format expected by the model's forward() method.
        
        Following mmdet3d's pattern, this method handles:
        - Stacking images (if they come as lists)
        - Stacking transformation matrices (if they come as lists)
        - Moving tensors to the correct device
        - Extracting GT annotations from data_samples

        Args:
            data (dict): The output of dataloader with 'inputs' and 'data_samples' keys.

        Returns:
            tuple: (batch_data, data_samples_list) where batch_data is a dict
                of model inputs and data_samples_list is a list of data samples.
        """
        inputs = data.get('inputs', {})
        data_samples = data.get('data_samples', [])
        
        # Get device from model
        device = next(self.parameters()).device
        
        # Extract inputs dict - move all tensors to the correct device
        # Handle stacking for images and transformation matrices (like mmdet3d's data preprocessor)
        batch_data = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.to(device)
            elif isinstance(value, list):
                # Check if it's a list of tensors that should be stacked
                if len(value) > 0 and isinstance(value[0], torch.Tensor):
                    # Only stack if all tensors have the same shape (can be batched)
                    # This handles images and transformation matrices that are already padded/aligned
                    # Point clouds and other variable-length data should remain as lists
                    first_shape = value[0].shape
                    all_same_shape = all(v.shape == first_shape for v in value if isinstance(v, torch.Tensor))
                    
                    if all_same_shape and 'points' not in key:
                        # Stack tensors (images, matrices, transformations, etc.) that have same shape
                        batch_data[key] = torch.stack([v.to(device) for v in value])
                    else:
                        # For point clouds and variable-length data, keep as list but move to device
                        batch_data[key] = [
                            v.to(device) if isinstance(v, torch.Tensor) else v
                            for v in value
                        ]
                else:
                    # Non-tensor lists (like metas)
                    batch_data[key] = value
            else:
                batch_data[key] = value
        
        # Extract GT from data_samples
        if isinstance(data_samples, list):
            data_samples_list = data_samples
        else:
            data_samples_list = [data_samples]
        
        if len(data_samples_list) > 0 and hasattr(data_samples_list[0], 'gt_instances_3d'):
            # Get GT bboxes and labels from data_samples
            gt_bboxes_3d_list = []
            gt_labels_3d_list = []
            for ds in data_samples_list:
                if hasattr(ds.gt_instances_3d, 'bboxes_3d') and ds.gt_instances_3d.bboxes_3d is not None:
                    gt_bboxes_3d_list.append(ds.gt_instances_3d.bboxes_3d)
                if hasattr(ds.gt_instances_3d, 'labels_3d') and ds.gt_instances_3d.labels_3d is not None:
                    gt_labels_3d_list.append(ds.gt_instances_3d.labels_3d)
            
            if len(gt_bboxes_3d_list) > 0:
                batch_data['gt_bboxes_3d'] = [
                    bbox.to(device) if isinstance(bbox, torch.Tensor) else bbox
                    for bbox in gt_bboxes_3d_list
                ]
            if len(gt_labels_3d_list) > 0:
                batch_data['gt_labels_3d'] = [
                    label.to(device) if isinstance(label, torch.Tensor) else label
                    for label in gt_labels_3d_list
                ]
        
        # Extract metas from data_samples metainfo
        metas_list = []
        for ds in data_samples_list:
            meta_dict = {}
            if hasattr(ds, 'metainfo'):
                for key in ds.metainfo_keys():
                    meta_dict[key] = ds.get(key, None)
            metas_list.append(meta_dict)
        batch_data['metas'] = metas_list
        
        return batch_data, data_samples_list

    def train_step(self, data, optim_wrapper):
        """The iteration step during training.
        
        This method uses forward() with mode='loss' to get losses,
        following mmdet3d's pattern. BaseModel's train_step will call
        forward() with mode='loss' automatically, but we override to
        handle data preprocessing.
        
        Args:
            data (dict): The output of dataloader with 'inputs' and 'data_samples' keys.
            optim_wrapper (:obj:`OptimWrapper`): The optimizer wrapper.
        
        Returns:
            dict: Dictionary of log variables for logging.
        """
        # Preprocess data from dataloader format to model input format
        batch_data, _ = self._preprocess_data(data)
        
        # Forward pass to get losses using mode='loss'
        with optim_wrapper.optim_context(self):
            losses = self.forward(batch_data, mode='loss')

        # BaseModel's train_step will handle loss parsing and optimization
        # But we need to return log_vars, so we parse losses here
        from collections import OrderedDict
        import torch.distributed as dist
        
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")
        
        loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)
        log_vars["loss"] = loss
        
        for loss_name, loss_value in log_vars.items():
            if isinstance(loss_value, torch.Tensor):
                if dist.is_available() and dist.is_initialized():
                    loss_value = loss_value.data.clone()
                    dist.all_reduce(loss_value.div_(dist.get_world_size()))
                log_vars[loss_name] = loss_value.item()
        
        # Update parameters
        optim_wrapper.update_params(loss)
        
        return log_vars

    def val_step(self, data, optim_wrapper=None):
        """The iteration step during validation.

        This method uses forward() with mode='predict' to get predictions,
        following mmdet3d's pattern.
        
        Args:
            data (dict): The output of dataloader with 'inputs' and 'data_samples' keys.
            optim_wrapper (:obj:`OptimWrapper`, optional): Unused, kept for compatibility.
        
        Returns:
            list[dict]: List of prediction dictionaries, one per sample.
        """
        # Set model to eval mode
        self.eval()
        
        # Preprocess data
        batch_data, data_samples_list = self._preprocess_data(data)
        
        # Forward pass to get predictions using mode='predict'
        with torch.no_grad():
            predictions = self.forward(batch_data, data_samples=data_samples_list, mode='predict')
        
        # Convert predictions to format expected by evaluator
        outputs = []
        for i, pred_dict in enumerate(predictions):
            output = {
                'boxes_3d': pred_dict.get('boxes_3d'),
                'scores_3d': pred_dict.get('scores_3d'),
                'labels_3d': pred_dict.get('labels_3d'),
            }
            if i < len(data_samples_list):
                if hasattr(data_samples_list[i], 'sample_idx'):
                    output['sample_idx'] = data_samples_list[i].sample_idx
                elif hasattr(data_samples_list[i], 'img_path'):
                    output['sample_idx'] = data_samples_list[i].img_path
            outputs.append(output)
        
        return outputs
