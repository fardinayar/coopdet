"""3D Transforms for cooperative 3D detection."""
from typing import Any, Dict

import numpy as np
import torch
import torchvision
from numpy import random
from PIL import Image

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import (
    CameraInstance3DBoxes,
    DepthInstance3DBoxes,
    LiDARInstance3DBoxes,
)
from mmdet3d.structures.ops import box_np_ops
from mmengine.registry import build_from_cfg

from ..builder import OBJECTSAMPLERS


@TRANSFORMS.register_module()
class ImageAug3DCoop:
    """Image augmentation for cooperative 3D detection."""

    def __init__(
        self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip, is_train
    ):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, results, vehicle):
        if vehicle:
            W, H = results["vehicle_ori_shape"]
        else:
            W, H = results["infrastructure_ori_shape"]

        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(
        self, img, rotation, translation, resize, resize_dims, crop, flip, rotate
    ):
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs_vehicle = data["vehicle_img"]
        imgs_infrastructure = data["infrastructure_img"]
        new_imgs_vehicle = []
        new_imgs_infrastructure = []
        transforms_vehicle = []
        transforms_infrastructure = []

        for img_vehicle in imgs_vehicle:
            resize_vehicle, resize_dims_vehicle, crop_vehicle, flip_vehicle, rotate_vehicle = self.sample_augmentation(
                data, True)
            post_rot_vehicle = torch.eye(2)
            post_tran_vehicle = torch.zeros(2)
            new_img_vehicle, rotation_vehicle, translation_vehicle = self.img_transform(
                img_vehicle,
                post_rot_vehicle,
                post_tran_vehicle,
                resize=resize_vehicle,
                resize_dims=resize_dims_vehicle,
                crop=crop_vehicle,
                flip=flip_vehicle,
                rotate=rotate_vehicle,
            )
            transform_vehicle = torch.eye(4)
            transform_vehicle[:2, :2] = rotation_vehicle
            transform_vehicle[:2, 3] = translation_vehicle
            new_imgs_vehicle.append(new_img_vehicle)
            transforms_vehicle.append(transform_vehicle.numpy())
        data["vehicle_img"] = new_imgs_vehicle
        data["vehicle_img_aug_matrix"] = transforms_vehicle

        for img_infrastructure in imgs_infrastructure:
            resize_infrastructure, resize_dims_infrastructure, crop_infrastructure, flip_infrastructure, rotate_infrastructure = self.sample_augmentation(
                data, False)
            post_rot_infrastructure = torch.eye(2)
            post_tran_infrastructure = torch.zeros(2)
            new_img_infrastructure, rotation_infrastructure, translation_infrastructure = self.img_transform(
                img_infrastructure,
                post_rot_infrastructure,
                post_tran_infrastructure,
                resize=resize_infrastructure,
                resize_dims=resize_dims_infrastructure,
                crop=crop_infrastructure,
                flip=flip_infrastructure,
                rotate=rotate_infrastructure,
            )
            transform_infrastructure = torch.eye(4)
            transform_infrastructure[:2, :2] = rotation_infrastructure
            transform_infrastructure[:2, 3] = translation_infrastructure
            new_imgs_infrastructure.append(new_img_infrastructure)
            transforms_infrastructure.append(transform_infrastructure.numpy())
        data["infrastructure_img"] = new_imgs_infrastructure
        data["infrastructure_img_aug_matrix"] = transforms_infrastructure

        return data


@TRANSFORMS.register_module()
class GlobalRotScaleTransCoop:
    """Global rotation, scaling and translation for cooperative perception."""

    def __init__(self, resize_lim, rot_lim, trans_lim, is_train):
        self.resize_lim = resize_lim
        self.rot_lim = rot_lim
        self.trans_lim = trans_lim
        self.is_train = is_train

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        transform = np.eye(4).astype(np.float32)

        if self.is_train:
            scale = random.uniform(*self.resize_lim)
            theta = random.uniform(*self.rot_lim)
            translation = np.array([random.normal(0, self.trans_lim) for i in range(3)])
            rotation = np.eye(3)

            data["vehicle_points"].rotate(-theta)
            data["vehicle_points"].translate(translation)
            data["vehicle_points"].scale(scale)

            data["infrastructure_points"].rotate(-theta)
            data["infrastructure_points"].translate(translation)
            data["infrastructure_points"].scale(scale)

            gt_boxes = data["gt_bboxes_3d"]
            # FIX: Original coopdet3d rotated points by -theta but boxes by +theta (BUG!)
            # This caused misalignment. We fix it by rotating boxes by -theta too.
            # The checkpoint was trained with the bug, so using LiDAR-only weights is recommended.
            rot_sin = np.sin(-theta)
            rot_cos = np.cos(-theta)
            rot_mat = np.array(
                [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
                dtype=np.float32
            )
            rotation = rotation @ rot_mat
            if gt_boxes is not None:
                gt_boxes.rotate(-theta)  # Fixed: use -theta to align with points
            gt_boxes.translate(translation)
            gt_boxes.scale(scale)
            data["gt_bboxes_3d"] = gt_boxes

            transform[:3, :3] = rotation.T * scale
            transform[:3, 3] = translation * scale

        data["vehicle_lidar_aug_matrix"] = transform
        data["infrastructure_lidar_aug_matrix"] = transform

        return data


@TRANSFORMS.register_module()
class VehiclePointsToInfraCoords:
    """Transform vehicle points to infrastructure coordinates.
    
    Note: GT boxes are already in infrastructure coordinates when loaded from the dataset,
    so they don't need to be transformed here.
    """

    def __call__(self, data):
        v2i = np.asarray(data["vehicle2infrastructure"])
        v2i_rot = v2i[:3, :3]
        v2i_trans = v2i[:3, 3]
        data["vehicle_points"].rotate(v2i_rot.T)
        data["vehicle_points"].translate(v2i_trans)
        return data


@TRANSFORMS.register_module()
class GridMaskCoop:
    """GridMask augmentation for cooperative perception."""

    def __init__(
        self,
        use_h,
        use_w,
        max_epoch,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=0,
        prob=1.0,
        fixed_prob=False,
    ):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.epoch = None
        self.max_epoch = max_epoch
        self.fixed_prob = fixed_prob

    def set_epoch(self, epoch):
        self.epoch = epoch
        if not self.fixed_prob:
            self.set_prob(self.epoch, self.max_epoch)

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * self.epoch / self.max_epoch

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        vehicle_imgs = results["vehicle_img"]
        infrastructure_imgs = results["infrastructure_img"]
        h1 = vehicle_imgs[0].shape[0]
        w1 = vehicle_imgs[0].shape[1]
        h2 = infrastructure_imgs[0].shape[0]
        w2 = infrastructure_imgs[0].shape[1]

        self.d1_vehicle = 2
        self.d2_vehicle = min(h1, w1)
        d_vehicle = np.random.randint(self.d1_vehicle, self.d2_vehicle)
        self.d1_infrastructure = 2
        self.d2_infrastructure = min(h2, w2)
        d_infrastructure = np.random.randint(self.d1_infrastructure, self.d2_infrastructure)

        hh1 = int(1.5 * h1)
        ww1 = int(1.5 * w1)
        hh2 = int(1.5 * h2)
        ww2 = int(1.5 * w2)

        if self.ratio == 1:
            self.l1 = np.random.randint(1, d_vehicle)
            self.l2 = np.random.randint(1, d_infrastructure)
        else:
            self.l1 = min(max(int(d_vehicle * self.ratio + 0.5), 1), d_vehicle - 1)
            self.l2 = min(max(int(d_infrastructure * self.ratio + 0.5), 1), d_infrastructure - 1)
        mask1 = np.ones((hh1, ww1), np.float32)
        mask2 = np.ones((hh2, ww2), np.float32)
        st_h1 = np.random.randint(d_vehicle)
        st_w1 = np.random.randint(d_vehicle)
        st_h2 = np.random.randint(d_infrastructure)
        st_w2 = np.random.randint(d_infrastructure)

        if self.use_h:
            for i in range(hh1 // d_vehicle):
                s1 = d_vehicle * i + st_h1
                t1 = min(s1 + self.l1, hh1)
                mask1[s1:t1, :] *= 0
            for j in range(hh2 // d_infrastructure):
                s2 = d_infrastructure * j + st_h2
                t2 = min(s2 + self.l2, hh2)
                mask2[s2:t2, :] *= 0

        if self.use_w:
            for i in range(ww1 // d_vehicle):
                s1 = d_vehicle * i + st_w1
                t1 = min(s1 + self.l1, ww1)
                mask1[:, s1:t1] *= 0
            for j in range(ww2 // d_infrastructure):
                s2 = d_infrastructure * j + st_w2
                t2 = min(s2 + self.l2, ww2)
                mask2[:, s2:t2] *= 0

        r1 = np.random.randint(self.rotate)
        r2 = np.random.randint(self.rotate)

        mask1 = Image.fromarray(np.uint8(mask1))
        mask1 = mask1.rotate(r1)
        mask1 = np.asarray(mask1)
        mask1 = mask1[
            (hh1 - h1) // 2: (hh1 - h1) // 2 + h1, (ww1 - w1) // 2: (ww1 - w1) // 2 + w1
        ]
        mask2 = Image.fromarray(np.uint8(mask2))
        mask2 = mask2.rotate(r2)
        mask2 = np.asarray(mask2)
        mask2 = mask2[
            (hh2 - h2) // 2: (hh2 - h2) // 2 + h2, (ww2 - w2) // 2: (ww2 - w2) // 2 + w2
        ]

        mask1 = mask1.astype(np.float32)
        mask1 = mask1[:, :, None]
        mask2 = mask2.astype(np.float32)
        mask2 = mask2[:, :, None]
        if self.mode == 1:
            mask1 = 1 - mask1
            mask2 = 1 - mask2

        if self.offset:
            offset1 = torch.from_numpy(2 * (np.random.rand(h1, w1) - 0.5)).float()
            offset2 = torch.from_numpy(2 * (np.random.rand(h2, w2) - 0.5)).float()
            offset1 = (1 - mask1) * offset1
            offset2 = (1 - mask2) * offset2
            vehicle_imgs = [x * mask1 + offset1 for x in vehicle_imgs]
            infrastructure_imgs = [x * mask2 + offset2 for x in infrastructure_imgs]
        else:
            vehicle_imgs = [x * mask1 for x in vehicle_imgs]
            infrastructure_imgs = [x * mask2 for x in infrastructure_imgs]

        results.update(vehicle_img=vehicle_imgs)
        results.update(infrastructure_img=infrastructure_imgs)
        return results


@TRANSFORMS.register_module()
class ObjectPasteCoop:
    """Sample GT objects to the data for cooperative perception."""

    def __init__(self, db_sampler, sample_2d=False, stop_epoch=None):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        if "type" not in db_sampler.keys():
            db_sampler["type"] = "DataBaseSampler"
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)
        self.epoch = -1
        self.stop_epoch = stop_epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes."""
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, data):
        """Call function to sample ground truth objects to the data."""
        if self.stop_epoch is not None and self.epoch >= self.stop_epoch:
            return data
        gt_bboxes_3d = data["gt_bboxes_3d"]
        gt_labels_3d = data["gt_labels_3d"]

        vehicle_points = data["vehicle_points"]
        infrastructure_points = data["infrastructure_points"]

        if self.sample_2d:
            img = data["img"]
            gt_bboxes_2d = data["gt_bboxes"]
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                gt_bboxes_2d=gt_bboxes_2d,
                img=img,
            )
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(), gt_labels_3d, img=None
            )

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict["gt_bboxes_3d"]
            sampled_points = sampled_dict["points"]
            sampled_gt_labels = sampled_dict["gt_labels_3d"]

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels], axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate([gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d])
            )

            infrastructure_points = self.remove_points_in_boxes(infrastructure_points, sampled_gt_bboxes_3d)
            infrastructure_points = infrastructure_points.cat([sampled_points, infrastructure_points])

            vehicle_points = self.remove_points_in_boxes(vehicle_points, sampled_gt_bboxes_3d)
            vehicle_points = vehicle_points.cat([sampled_points, vehicle_points])

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict["gt_bboxes_2d"]
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]
                ).astype(np.float32)
                data["gt_bboxes"] = gt_bboxes_2d
                data["img"] = sampled_dict["img"]

        data["gt_bboxes_3d"] = gt_bboxes_3d
        data["gt_labels_3d"] = gt_labels_3d.astype(int)
        data["vehicle_points"] = vehicle_points
        data["infrastructure_points"] = infrastructure_points

        return data


@TRANSFORMS.register_module()
class PointShuffleCoop:
    """Shuffle points for cooperative perception."""

    def __call__(self, data):
        data["vehicle_points"].shuffle()
        data["infrastructure_points"].shuffle()
        return data


# Note: ObjectRangeFilter is already registered in mmdet3d.datasets.transforms.transforms_3d
# Import it from there if needed: from mmdet3d.datasets.transforms import ObjectRangeFilter

@TRANSFORMS.register_module()
class PointsRangeFilterCoop:
    """Filter points by the range for cooperative perception."""

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, data):
        """Call function to filter points by the range."""
        if "vehicle_points" in data:
            points = data["vehicle_points"]
            points_mask = points.in_range_3d(self.pcd_range)
            clean_points = points[points_mask]
            data["vehicle_points"] = clean_points

        if "infrastructure_points" in data:
            points = data["infrastructure_points"]
            points_mask = points.in_range_3d(self.pcd_range)
            clean_points = points[points_mask]
            data["infrastructure_points"] = clean_points
        return data


# Note: ObjectNameFilter is already registered in mmdet3d.datasets.transforms.transforms_3d
# Import it from there if needed: from mmdet3d.datasets.transforms import ObjectNameFilter

@TRANSFORMS.register_module()
class ImageNormalizeCoop:
    """Normalize images for cooperative perception."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["vehicle_img"] = [self.compose(img) for img in data["vehicle_img"]]
        data["infrastructure_img"] = [self.compose(img) for img in data["infrastructure_img"]]
        data["img_norm_cfg"] = dict(mean=self.mean, std=self.std)
        return data


__all__ = [
    'ImageAug3DCoop',
    'GlobalRotScaleTransCoop',
    'VehiclePointsToInfraCoords',
    'GridMaskCoop',
    'ObjectPasteCoop',
    'PointShuffleCoop',
    'PointsRangeFilterCoop',
    'ImageNormalizeCoop',
]

