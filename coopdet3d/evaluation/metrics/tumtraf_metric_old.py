# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Union

import mmengine
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.registry import METRICS as MMENGINE_METRICS

from mmdet3d.registry import METRICS


# Register in both mmdet3d and mmengine registries to ensure it's found
@METRICS.register_module()
@MMENGINE_METRICS.register_module()
class TUMTrafMetric(BaseMetric):
    """TUMTraf evaluation metric using nuScenes-like protocol.

    This evaluator uses the evaluation functions from the dataset class
    to compute metrics in the TUMTraf/nuScenes format.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'bbox'.
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_camera=False, use_lidar=True).
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result to a
            specific format and submit it to the test server.
            Defaults to False.
        jsonfile_prefix (str, optional): The prefix of json files including the
            file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        result_names (list[str]): Result names, usually equals to
            ['pts_bbox']. Defaults to ['pts_bbox'].
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 modality: dict = dict(use_camera=False, use_lidar=True),
                 prefix: Optional[str] = None,
                 format_only: bool = False,
                 jsonfile_prefix: Optional[str] = None,
                 result_names: List[str] = ['pts_bbox'],
                 dataset_type: Optional[str] = None,
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None) -> None:
        self.default_prefix = 'TUMTraf metric'
        super(TUMTrafMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)
        
        self.ann_file = ann_file
        self.data_root = data_root
        self.modality = modality
        self.format_only = format_only
        if self.format_only:
            assert jsonfile_prefix is not None, \
                'jsonfile_prefix must be not None when format_only is True, ' \
                'otherwise the result files will be saved to a temp directory ' \
                'which will be cleanup at the end.'

        self.jsonfile_prefix = jsonfile_prefix
        self.backend_args = backend_args
        self.result_names = result_names
        self.metrics = metric if isinstance(metric, list) else [metric]
        self.dataset_type = dataset_type

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
                The model returns a list of dicts with 'boxes_3d', 'scores_3d', 'labels_3d'.
                Or data_samples may contain 'pred_instances_3d' and 'sample_idx'.
        """
        # Handle case where model returns list of dicts directly
        if len(data_samples) > 0 and isinstance(data_samples[0], dict):
            # Check if it's the model output format (list of dicts with boxes_3d)
            if 'boxes_3d' in data_samples[0] or 'pred_instances_3d' in data_samples[0]:
                for data_sample in data_samples:
                    result = dict()
                    # Handle pred_instances_3d format (from mmengine runner)
                    if 'pred_instances_3d' in data_sample:
                        pred_3d = data_sample['pred_instances_3d']
                        if 'bboxes_3d' in pred_3d:
                            result['boxes_3d'] = pred_3d['bboxes_3d'].to('cpu') if hasattr(pred_3d['bboxes_3d'], 'to') else pred_3d['bboxes_3d']
                        if 'scores' in pred_3d:
                            scores = pred_3d['scores']
                            result['scores_3d'] = scores.cpu() if hasattr(scores, 'cpu') else scores
                        if 'labels' in pred_3d:
                            labels = pred_3d['labels']
                            result['labels_3d'] = labels.cpu() if hasattr(labels, 'cpu') else labels
                    # Handle direct boxes_3d format (from our model)
                    elif 'boxes_3d' in data_sample:
                        boxes = data_sample['boxes_3d']
                        result['boxes_3d'] = boxes.to('cpu') if hasattr(boxes, 'to') else boxes
                        if 'scores_3d' in data_sample:
                            scores = data_sample['scores_3d']
                            result['scores_3d'] = scores.cpu() if hasattr(scores, 'cpu') else scores
                        if 'labels_3d' in data_sample:
                            labels = data_sample['labels_3d']
                            result['labels_3d'] = labels.cpu() if hasattr(labels, 'cpu') else labels
                    
                    # Store sample index
                    if 'sample_idx' in data_sample:
                        result['sample_idx'] = data_sample['sample_idx']
                    elif 'img_path' in data_sample:
                        result['sample_idx'] = data_sample.get('img_path', '')
                    
                    self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # Get dataset metadata
        classes = self.dataset_meta['classes']
        
        # Import dataset classes directly to avoid registry scope issues
        # We'll need to access the dataset instance, but since we don't have it,
        # we'll create a temporary dataset instance just for evaluation
        from coopdet3d.datasets import TUMTrafV2XNuscDataset, TUMTrafNuscDataset
        
        # Determine dataset type from parameter, metadata, or infer from ann_file
        dataset_type = self.dataset_type
        if dataset_type is None:
            dataset_type = self.dataset_meta.get('dataset_type', None)
        if dataset_type is None:
            # Infer from ann_file - if it contains 'v2x', use TUMTrafV2XNuscDataset
            if 'v2x' in self.ann_file.lower():
                dataset_type = 'TUMTrafV2XNuscDataset'
            else:
                dataset_type = 'TUMTrafNuscDataset'
        
        # Create a temporary dataset instance for evaluation
        # We need the dataset to access format_results and _evaluate_single
        # Directly instantiate the class to avoid registry scope issues
        dataset_cfg = dict(
            data_root=self.data_root,
            ann_file=self.ann_file,
            modality=self.modality,
            test_mode=True,
            pipeline=[],
        )
        # Directly instantiate the dataset class instead of using registry
        if dataset_type == 'TUMTrafV2XNuscDataset':
            dataset = TUMTrafV2XNuscDataset(**dataset_cfg)
        else:
            dataset = TUMTrafNuscDataset(**dataset_cfg)
        
        # Format results using dataset's format_results method
        result_files, tmp_dir = dataset.format_results(
            results, jsonfile_prefix=self.jsonfile_prefix)

        metric_dict = {}

        if self.format_only:
            logger.info(
                f'results are saved in {osp.basename(self.jsonfile_prefix)}')
            if tmp_dir is not None:
                tmp_dir.cleanup()
            return metric_dict

        # Evaluate using dataset's _evaluate_single method
        for metric in self.metrics:
            if isinstance(result_files, dict):
                for name in self.result_names:
                    logger.info(f"Evaluating bboxes of {name}")
                    ret_dict = dataset._evaluate_single(
                        result_files[name],
                        logger=logger,
                        metric=metric,
                        result_name=name
                    )
                    metric_dict.update(ret_dict)
            elif isinstance(result_files, str):
                ret_dict = dataset._evaluate_single(
                    result_files,
                    logger=logger,
                    metric=metric,
                    result_name=self.result_names[0] if self.result_names else 'pts_bbox'
                )
                metric_dict.update(ret_dict)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        
        return metric_dict

