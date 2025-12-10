import torch
from mmdet3d.registry import TASK_UTILS


@TASK_UTILS.register_module()
class FocalLossCost:
    """FocalLossCost.
    Args:
        weight (int | float, optional): loss_weight
        alpha (int | float, optional): focal_loss alpha
        gamma (int | float, optional): focal_loss gamma
        eps (float, optional): default 1e-12
        binary_input (bool, optional): Whether the input is binary,
            default False.
    """

    def __init__(self,
                 weight=1.,
                 alpha=0.25,
                 gamma=2,
                 eps=1e-12,
                 binary_input=False):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.binary_input = binary_input

    def _focal_loss_cost(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_query, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_query, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
        Returns:
            torch.Tensor: cls_cost value with weight
        """
        return self._focal_loss_cost(cls_pred, gt_labels)

