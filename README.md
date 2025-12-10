# CoopDet3D: Cooperative 3D Object Detection
## TUMTraf V2X Dataset with mmengine

Multi-modal cooperative 3D object detection for vehicle and infrastructure sensors using camera, LiDAR, and fusion approaches.

## Features
- **3 Operating Modes**: Cooperative, Vehicle-only, Infrastructure-only
- **3 Modalities**: Camera-only, LiDAR-only, Camera-LiDAR Fusion
- **9 Configurations**: All combinations of modes × modalities
- **Metrics**: NuScenes mAP, BEV mAP, 3D mAP (IoU-based)

## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/coopdet3d
cd coopdet3d

# Install dependencies (Python 3.8-3.10)
pip install -r requirements.txt
python setup.py develop
```

## Dataset Setup

1. Download [TUMTraf V2X Dataset](https://innovation-mobility.com/tumtraf-dataset)
2. Prepare data:
```bash
python tools/create_tumtraf_v2x_data.py \
  --root-path data/tumtraf_v2x_cooperative_perception_dataset \
  --out-dir data/tumtraf_v2x_processed \
  --splits training,validation
```

Expected structure:
```
coopdet3d/
├── data/
│   └── tumtraf_v2x_processed/
│       ├── tumtraf_v2x_nusc_infos_train.pkl
│       ├── tumtraf_v2x_nusc_infos_val.pkl
│       └── tumtraf_v2x_nusc_gt_database/
├── weights/
│   └── yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1_new.pth
└── configs/
```

## Pretrained Weights

Download YOLOv8-s backbone (required for camera/fusion models):
```bash
mkdir -p weights
# Download from: https://github.com/open-mmlab/mmyolo/releases
# Place: weights/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1_new.pth
```

## Training Commands

### Cooperative (Vehicle + Infrastructure)

**Fusion (Camera + LiDAR)**
```bash
python tools/train_coop.py configs/cooperative_fusion.py --launcher pytorch
```
*Requires*: YOLOv8 weights (specified in config)

**LiDAR-only**
```bash
python tools/train_coop.py configs/cooperative_lidar.py --launcher pytorch
```
*Requires*: None

**Camera-only**
```bash
python tools/train_coop.py configs/cooperative_camera.py --launcher pytorch
```
*Requires*: YOLOv8 weights

### Vehicle-only

**Fusion (Camera + LiDAR)**
```bash
python tools/train.py configs/vehicle_fusion.py --launcher pytorch
```
*Requires*: YOLOv8 weights

**LiDAR-only**
```bash
python tools/train.py configs/vehicle_lidar.py --launcher pytorch
```
*Requires*: None

**Camera-only**
```bash
python tools/train.py configs/vehicle_camera.py --launcher pytorch
```
*Requires*: YOLOv8 weights

### Infrastructure-only

**Fusion (Camera + LiDAR)**
```bash
python tools/train.py configs/infrastructure_fusion.py --launcher pytorch
```
*Requires*: YOLOv8 weights

**LiDAR-only**
```bash
python tools/train.py configs/infrastructure_lidar.py --launcher pytorch
```
*Requires*: None

**Camera-only**
```bash
python tools/train.py configs/infrastructure_camera.py --launcher pytorch
```
*Requires*: YOLOv8 weights

## Testing

**Cooperative models**:
```bash
python tools/test_coop.py configs/cooperative_fusion.py checkpoints/epoch_20.pth
```

**Single-agent models**:
```bash
python tools/test.py configs/vehicle_lidar.py checkpoints/epoch_20.pth
```

## Evaluation Metrics

Models are evaluated with:
- **NuScenes mAP**: Center distance-based matching (0.5m, 1m, 2m, 4m)
- **BEV mAP**: IoU-based matching in bird's eye view (IoU 0.5, 0.7)
- **3D mAP**: Full 3D IoU matching (IoU 0.5, 0.7)
- **TP Errors**: mATE, mASE, mAOE, mAVE
- **NDS**: NuScenes Detection Score

Example output:
```
NuScenes mAP: 0.2946  |  BEV mAP: 0.1138  |  3D mAP: 0.0719
```

## Configuration Matrix

| Config | Mode | Modality | Tool | Weights Needed |
|--------|------|----------|------|----------------|
| `cooperative_fusion.py` | Cooperative | Cam+LiDAR | `train_coop.py` | YOLOv8 |
| `cooperative_lidar.py` | Cooperative | LiDAR | `train_coop.py` | - |
| `cooperative_camera.py` | Cooperative | Camera | `train_coop.py` | YOLOv8 |
| `vehicle_fusion.py` | Vehicle | Cam+LiDAR | `train.py` | YOLOv8 |
| `vehicle_lidar.py` | Vehicle | LiDAR | `train.py` | - |
| `vehicle_camera.py` | Vehicle | Camera | `train.py` | YOLOv8 |
| `infrastructure_fusion.py` | Infrastructure | Cam+LiDAR | `train.py` | YOLOv8 |
| `infrastructure_lidar.py` | Infrastructure | LiDAR | `train.py` | - |
| `infrastructure_camera.py` | Infrastructure | Camera | `train.py` | YOLOv8 |

## Quick Start Example

```bash
# 1. Download and prepare dataset
python tools/create_tumtraf_v2x_data.py --root-path data/tumtraf_v2x_cooperative_perception_dataset --out-dir data/tumtraf_v2x_processed --splits training,validation

# 2. Download YOLOv8 weights to weights/

# 3. Train cooperative fusion model
python tools/train_coop.py configs/cooperative_fusion.py --launcher pytorch

# 4. Test trained model
python tools/test_coop.py configs/cooperative_fusion.py work_dirs/cooperative_fusion/epoch_20.pth
```

## Advanced Options

**Multi-GPU training**:
```bash
python tools/train_coop.py configs/cooperative_fusion.py --launcher pytorch
# Automatically uses all available GPUs
```

**Resume training**:
```bash
python tools/train_coop.py configs/cooperative_fusion.py --resume
```

**Override config**:
```bash
python tools/train_coop.py configs/cooperative_fusion.py \
  --cfg-options train_cfg.max_epochs=30 optim_wrapper.optimizer.lr=0.0002
```

**Load pretrained weights**:
```bash
python tools/train.py configs/vehicle_lidar.py --load-from pretrained_lidar.pth
```

## Citation
```bibtex
@inproceedings{zimmer2024tumtrafv2x,
  title={TUMTraf V2X Cooperative Perception Dataset},
  author={Zimmer, Walter and Wardana, Gerhard Arya and Sritharan, Suren and Zhou, Xingcheng and Song, Rui and Knoll, Alois C.},
  booktitle={CVPR},
  year={2024}
}
```

## Acknowledgments
Built on [BEVFusion](https://github.com/mit-han-lab/bevfusion), [mmdet3d](https://github.com/open-mmlab/mmdetection3d), and [mmengine](https://github.com/open-mmlab/mmengine).

## License
MIT License. See LICENSE file for details.
