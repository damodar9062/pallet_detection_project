# Pallet Detection Project

## Overview
This project implements pallet detection and ground segmentation using YOLOv8 and DeepLabV3 in ROS2 Humble on Ubuntu 22.04.

## Metrics
- YOLOv8 Test mAP: 0.1246 (in progress, retraining to improve)
- DeepLabV3 Validation IoU: 0.8915
- DeepLabV3 Test IoU: [Pending evaluation]

## Files
- `augment.py`: Data augmentation for training.
- `train_yolo.py`: Trains YOLOv8 model.
- `train_deeplab.py`: Trains DeepLabV3 model.
- `evaluate_yolo.py`: Evaluates YOLOv8 mAP.
- `evaluate_deeplab.py`: Evaluates DeepLabV3 IoU.
- `convert_txt_to_png.py`: Converts YOLO .txt labels to .png masks.
- `pallet_detection_node.py`: ROS2 node for pallet detection and ground segmentation.
- `publisher_node.py`: ROS2 node to publish test images.
- `pallet_detection/data.yaml`: YOLO dataset configuration.
- `requirements.txt`: Python dependencies.

## Models
- YOLOv8: [Google Drive link to best.onnx]
- DeepLabV3: [Google Drive link to deeplabv3.pth]

## Setup
1. Install ROS2 Humble: https://docs.ros.org/en/humble/Installation.html
2. Clone this repository: `git clone https://github.com/damodar9062/pallet_detection_project.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Build ROS2 packages: `cd ~/ros2_ws && colcon build`
5. Source workspace: `source ~/ros2_ws/install/setup.bash`
6. Run nodes:
   - `ros2 run pallet_detection pallet_detection_node`
   - `ros2 run custom_image_publisher image_publisher_node`
   - `ros2 run image_view image_view --ros-args -r image:=/pallet_detection`
   - `ros2 run image_view image_view --ros-args -r image:=/ground_segmentation`

## Usage
- Train YOLOv8: `python3 train_yolo.py`
- Train DeepLabV3: `python3 train_deeplab.py`
- Evaluate: `python3 evaluate_yolo.py` and `python3 evaluate_deeplab.py`