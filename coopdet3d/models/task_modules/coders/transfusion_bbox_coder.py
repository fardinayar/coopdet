import torch

from mmdet.models.task_modules import BaseBBoxCoder
from mmdet3d.registry import TASK_UTILS


@TASK_UTILS.register_module()
class TransFusionBBoxCoder(BaseBBoxCoder):
    def __init__(self,
                 pc_range,
                 out_size_factor,
                 voxel_size,
                 post_center_range=None,
                 score_threshold=None,
                 code_size=8,
                 ):
        # Convert to lists/tuples if they are tensors to allow indexing
        if isinstance(pc_range, torch.Tensor):
            self.pc_range = pc_range.tolist()
        else:
            self.pc_range = list(pc_range) if not isinstance(pc_range, (list, tuple)) else pc_range
        
        if isinstance(voxel_size, torch.Tensor):
            self.voxel_size = voxel_size.tolist()
        else:
            self.voxel_size = list(voxel_size) if not isinstance(voxel_size, (list, tuple)) else voxel_size
        
        self.out_size_factor = out_size_factor
        self.post_center_range = post_center_range
        self.score_threshold = score_threshold
        self.code_size = code_size

    def encode(self, dst_boxes):
        targets = torch.zeros([dst_boxes.shape[0], self.code_size]).to(dst_boxes.device)
        # Extract scalar values safely
        pc_range_0 = float(self.pc_range[0])
        pc_range_1 = float(self.pc_range[1])
        voxel_size_0 = float(self.voxel_size[0])
        voxel_size_1 = float(self.voxel_size[1])
        
        targets[:, 0] = (dst_boxes[:, 0] - pc_range_0) / (self.out_size_factor * voxel_size_0)
        targets[:, 1] = (dst_boxes[:, 1] - pc_range_1) / (self.out_size_factor * voxel_size_1)
        # targets[:, 2] = (dst_boxes[:, 2] - self.post_center_range[2]) / (self.post_center_range[5] - self.post_center_range[2])
        targets[:, 3] = dst_boxes[:, 3].log()
        targets[:, 4] = dst_boxes[:, 4].log()
        targets[:, 5] = dst_boxes[:, 5].log()
        targets[:, 2] = dst_boxes[:, 2] + dst_boxes[:, 5] * 0.5  # bottom center to gravity center
        targets[:, 6] = torch.sin(dst_boxes[:, 6])
        targets[:, 7] = torch.cos(dst_boxes[:, 6])
        if self.code_size == 10:
            targets[:, 8:10] = dst_boxes[:, 7:]
        return targets

    def decode(self, heatmap, rot, dim, center, height, vel, filter=False):
        """Decode bboxes.
        Args:
            heatmap (torch.Tensor): Heatmap with the shape of [B, num_cls, num_proposals].
            rot (torch.Tensor): Rotation (sin, cos) with the shape of
                [B, 2, num_proposals].
            dim (torch.Tensor): Dim of the boxes (log-encoded) with the shape of
                [B, 3, num_proposals].
            center (torch.Tensor): bev center (x, y) of the boxes with the shape of
                [B, 2, num_proposals]. (in feature map metric)
            height (torch.Tensor): height (z, gravity center) of the boxes with the shape of
                [B, 1, num_proposals]. (in real world metric)
            vel (torch.Tensor): Velocity with the shape of [B, 2, num_proposals].
            filter: if False, return all box without checking score and center_range
        Returns:
            list[dict]: Decoded boxes.
        """
        # class label
        final_preds = heatmap.max(1, keepdims=False).indices
        final_scores = heatmap.max(1, keepdims=False).values

        # change size to real world metric
        # Extract scalar values (already converted to lists in __init__)
        voxel_size_0 = float(self.voxel_size[0])
        voxel_size_1 = float(self.voxel_size[1])
        pc_range_0 = float(self.pc_range[0])
        pc_range_1 = float(self.pc_range[1])
        
        center[:, 0, :] = center[:, 0, :] * self.out_size_factor * voxel_size_0 + pc_range_0
        center[:, 1, :] = center[:, 1, :] * self.out_size_factor * voxel_size_1 + pc_range_1
        # center[:, 2, :] = center[:, 2, :] * (self.post_center_range[5] - self.post_center_range[2]) + self.post_center_range[2]
        dim[:, 0, :] = dim[:, 0, :].exp()
        dim[:, 1, :] = dim[:, 1, :].exp()
        dim[:, 2, :] = dim[:, 2, :].exp()
        height = height - dim[:, 2:3, :] * 0.5  # gravity center to bottom center
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)

        if vel is None:
            final_box_preds = torch.cat([center, height, dim, rot], dim=1).permute(0, 2, 1)
        else:
            final_box_preds = torch.cat([center, height, dim, rot, vel], dim=1).permute(0, 2, 1)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            boxes3d = final_box_preds[i]
            scores = final_scores[i]
            labels = final_preds[i]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }
            predictions_dicts.append(predictions_dict)

        if filter is False:
            return predictions_dicts

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:
            # Convert to tensor if not already, avoiding copy construction warning
            if not isinstance(self.post_center_range, torch.Tensor):
                self.post_center_range = torch.tensor(
                        self.post_center_range, device=heatmap.device, dtype=torch.float32)
            else:
                self.post_center_range = self.post_center_range.to(device=heatmap.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(2)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(2)

            predictions_dicts = []
            for i in range(heatmap.shape[0]):
                cmask = mask[i, :]
                num_before_score = cmask.sum().item()
                if self.score_threshold:
                    cmask &= thresh_mask[i]
                
                num_after_score = cmask.sum().item()
                num_filtered_score = num_before_score - num_after_score

                boxes3d = final_box_preds[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]
                
                # DEBUG: Log filtering statistics (first batch, first evaluation only)
                if not hasattr(self, '_debug_filter_logged') and i == 0:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning("=" * 60)
                    logger.warning("BBox Coder Filtering Statistics (first eval batch):")
                    logger.warning(f"  Batch {i}:")
                    logger.warning(f"    Total predictions before filtering: {len(final_box_preds[i])}")
                    logger.warning(f"    Post-center-range filter: {num_before_score} predictions")
                    if self.score_threshold:
                        logger.warning(f"    Score threshold ({self.score_threshold}): filtered {num_filtered_score}, kept {num_after_score}")
                    logger.warning(f"    Final predictions after filtering: {len(boxes3d)}")
                    if len(boxes3d) == 0:
                        logger.warning(f"    WARNING: All predictions filtered out!")
                        logger.warning(f"    Score range before filter: [{final_scores[i].min():.4f}, {final_scores[i].max():.4f}]")
                        logger.warning(f"    Center range before filter: x=[{final_box_preds[i, :, 0].min():.2f}, {final_box_preds[i, :, 0].max():.2f}], y=[{final_box_preds[i, :, 1].min():.2f}, {final_box_preds[i, :, 1].max():.2f}], z=[{final_box_preds[i, :, 2].min():.2f}, {final_box_preds[i, :, 2].max():.2f}]")
                        logger.warning(f"    Post-center-range: {self.post_center_range.tolist()}")
                    logger.warning("=" * 60)
                    self._debug_filter_logged = True
                
                predictions_dict = {
                    'bboxes': boxes3d,
                    'scores': scores,
                    'labels': labels
                }

                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        return predictions_dicts

