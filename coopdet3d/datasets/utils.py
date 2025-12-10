import mmcv
import numpy as np


def extract_result_dict(results, key):
    """Extract and return the data corresponding to key in result dict.

    ``results`` is a dict output from `pipeline(input_dict)`, which is the
        loaded data from ``Dataset`` class.
    The data terms inside may be wrapped in list, tuple and DataContainer, so
        this function essentially extracts data from these wrappers.

    Args:
        results (dict): Data loaded using pipeline.
        key (str): Key of the desired data.

    Returns:
        np.ndarray | torch.Tensor | None: Data term.
    """
    if key not in results.keys():
        return None
    # results[key] may be data or list[data] or tuple[data]
    # data may be wrapped inside DataContainer
    data = results[key]
    if isinstance(data, (list, tuple)):
        data = data[0]
    if isinstance(data, mmcv.parallel.DataContainer):
        data = data._data
    return data


def output_to_box_dict(detection):
    """Convert detection output to box dictionary format.

    Args:
        detection (dict): Detection results with:
            - boxes_3d: BaseInstance3DBoxes with detection bboxes
            - scores_3d: torch.Tensor with detection scores
            - labels_3d: torch.Tensor with predicted labels

    Returns:
        list[dict]: List of box dicts with keys: center, wlh, orientation,
            label, score, velocity, name.
    """
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    box_list = []
    for i in range(len(box3d)):
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        box = {
            "center": np.array(box_gravity_center[i]),
            "wlh": np.array(box_dims[i]),
            "orientation": box_yaw[i],
            "label": int(labels[i]) if not np.isnan(labels[i]) else labels[i],
            "score": float(scores[i]) if not np.isnan(scores[i]) else scores[i],
            "velocity": np.array(velocity),
            "name": None
        }
        box_list.append(box)
    return box_list


def filter_box_in_lidar_cs(boxes, classes, eval_configs):
    """Filter boxes by detection range in LiDAR coordinate system.

    Args:
        boxes (list[dict]): List of predicted box dicts.
        classes (list[str]): Mapped class names.
        eval_configs (dict): Evaluation config with 'class_range' key.

    Returns:
        list[dict]: Filtered list of box dicts.
    """
    box_list = []
    for box in boxes:
        cls_range_map = eval_configs["class_range"]
        radius = np.linalg.norm(box["center"][:2], 2)
        det_range = cls_range_map[classes[box["label"]]]
        if radius > det_range:
            continue
        box_list.append(box)
    return box_list
