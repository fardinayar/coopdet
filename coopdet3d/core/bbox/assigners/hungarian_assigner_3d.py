# modify from https://github.com/mit-han-lab/bevfusion and mmdet3d projects/BEVFusion
import torch
from mmdet.models.task_modules import AssignResult, BaseAssigner
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment

from mmdet3d.registry import TASK_UTILS


@TASK_UTILS.register_module()
class BBoxBEVL1Cost:
    """BBoxBEVL1Cost for 3D detection.

    Args:
        weight (float): Weight of the cost.
    """

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, bboxes, gt_bboxes, train_cfg):
        """Compute BEV L1 cost between bboxes and gt_bboxes.

        Args:
            bboxes (torch.Tensor): Predicted boxes with shape (num_queries, 7+).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (num_gts, 7+).
            train_cfg (dict): Training config.

        Returns:
            torch.Tensor: Cost matrix with shape (num_queries, num_gts).
        """
        pc_start = bboxes.new(train_cfg['point_cloud_range'][0:2])
        pc_range = bboxes.new(train_cfg['point_cloud_range'][3:5]) - bboxes.new(train_cfg['point_cloud_range'][0:2])
        # normalize the box center to [0, 1]
        normalized_bboxes_xy = (bboxes[:, :2] - pc_start) / pc_range
        normalized_gt_bboxes_xy = (gt_bboxes[:, :2] - pc_start) / pc_range
        reg_cost = torch.cdist(normalized_bboxes_xy, normalized_gt_bboxes_xy, p=1)
        return reg_cost * self.weight


@TASK_UTILS.register_module()
class IoU3DCost:
    """IoU3DCost for 3D detection.

    Args:
        weight (float): Weight of the cost.
    """

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, iou):
        """Compute IoU cost from IoU matrix.

        Args:
            iou (torch.Tensor): IoU matrix with shape (num_queries, num_gts).

        Returns:
            torch.Tensor: Cost matrix with shape (num_queries, num_gts).
        """
        iou_cost = -iou
        return iou_cost * self.weight


@TASK_UTILS.register_module()
class HungarianAssigner3D(BaseAssigner):
    """Hungarian Assigner for 3D detection.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost.

    Args:
        cls_cost (dict): Config of classification cost.
        reg_cost (dict): Config of regression cost.
        iou_cost (dict): Config of IoU cost.
        iou_calculator (dict): Config of IoU calculator.
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxBEVL1Cost', weight=1.0),
                 iou_cost=dict(type='IoU3DCost', weight=1.0),
                 iou_calculator=dict(type='BboxOverlaps3D')):
        self.cls_cost = TASK_UTILS.build(cls_cost)
        self.reg_cost = TASK_UTILS.build(reg_cost)
        self.iou_cost = TASK_UTILS.build(iou_cost)
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(self, bboxes, gt_bboxes, gt_labels, cls_pred, train_cfg):
        """Computes one-to-one matching based on the weighted costs.

        Args:
            bboxes (torch.Tensor): Predicted boxes with shape (num_queries, 7+).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (num_gts, 7+).
            gt_labels (torch.Tensor): Ground truth labels with shape (num_gts,).
            cls_pred (torch.Tensor): Predicted classification logits with shape
                (num_decoder_layers, num_classes, num_queries) or similar.
            train_cfg (dict): Training config.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        cls_cost = self.cls_cost(cls_pred[0].T, gt_labels)
        reg_cost = self.reg_cost(bboxes, gt_bboxes, train_cfg)
        iou = self.iou_calculator(bboxes, gt_bboxes)
        iou_cost = self.iou_cost(iou)

        # weighted sum of above three costs
        cost = cls_cost + reg_cost + iou_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bboxes.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bboxes.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        max_overlaps = torch.zeros_like(iou.max(1).values)
        max_overlaps[matched_row_inds] = iou[matched_row_inds, matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)


@TASK_UTILS.register_module()
class HeuristicAssigner3D(BaseAssigner):
    """Heuristic Assigner for 3D detection.

    Args:
        dist_thre (float): Distance threshold for assignment.
        iou_calculator (dict): Config of IoU calculator.
    """

    def __init__(self,
                 dist_thre=100,
                 iou_calculator=dict(type='BboxOverlaps3D')):
        self.dist_thre = dist_thre  # distance in meter
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None, query_labels=None):
        """Assign ground truth to predictions based on distance.

        Args:
            bboxes (torch.Tensor): Predicted boxes with shape (num_queries, 7+).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (num_gts, 7+).
            gt_bboxes_ignore (torch.Tensor, optional): Ignored ground truth boxes.
            gt_labels (torch.Tensor, optional): Ground truth labels.
            query_labels (torch.Tensor, optional): Query labels.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        dist_thre = self.dist_thre
        num_gts, num_bboxes = len(gt_bboxes), len(bboxes)

        bev_dist = torch.norm(
            bboxes[:, 0:2][None, :, :] - gt_bboxes[:, 0:2][:, None, :],
            dim=-1)  # [num_gts, num_bboxes]
        if query_labels is not None:
            # only match the gt box and query with same category
            not_same_class = (query_labels[None] != gt_labels[:, None])
            bev_dist += not_same_class * dist_thre

        # for each gt box, assign it to the nearest pred box
        nearest_values, nearest_indices = bev_dist.min(1)  # [num_gts]
        assigned_gt_inds = torch.ones([num_bboxes, ]).to(bboxes) * 0
        assigned_gt_vals = torch.ones([num_bboxes, ]).to(bboxes) * 10000
        assigned_gt_labels = torch.ones([num_bboxes, ]).to(bboxes) * -1
        for idx_gts in range(num_gts):
            idx_pred = nearest_indices[idx_gts]  # each gt only match to the nearest pred box
            if bev_dist[idx_gts, idx_pred] <= dist_thre:
                if bev_dist[idx_gts, idx_pred] < assigned_gt_vals[idx_pred]:  # if this pred box is assigned, then compare
                    assigned_gt_vals[idx_pred] = bev_dist[idx_gts, idx_pred]
                    assigned_gt_inds[idx_pred] = idx_gts + 1  # for AssignResult, 0 is negative, -1 is ignore, 1-based indices are positive
                    assigned_gt_labels[idx_pred] = gt_labels[idx_gts]

        max_overlaps = torch.zeros([num_bboxes, ]).to(bboxes)
        matched_indices = torch.where(assigned_gt_inds > 0)
        matched_iou = self.iou_calculator(
            gt_bboxes[assigned_gt_inds[matched_indices].long() - 1],
            bboxes[matched_indices]).diag()
        max_overlaps[matched_indices] = matched_iou
        return AssignResult(
            num_gts, assigned_gt_inds.long(), max_overlaps, labels=assigned_gt_labels)

