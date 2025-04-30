# Pallet Detection and Ground Segmentation System

## Overview
This project implements real-time pallet detection and ground segmentation for robotic warehouse navigation using ROS2 Humble, YOLOv8, and DeepLabV3 on Ubuntu 22.04. It processes live camera data from a ZED 2i camera and publishes results on `/pallet_detection` (bounding boxes) and `/ground_segmentation` (masks) for evaluation on an NVIDIA AGX Orin.

## Metrics
- YOLOv8 Test mAP: 0.3446 (retraining in progress to improve accuracy)
- DeepLabV3 Validation IoU: 0.8915
- DeepLabV3 Test IoU: 0.4603

## Repository Contents
- `README.md`: Project documentation and setup instructions.
- `requirements.txt`: Python dependencies for training and inference.
- `train_yolo.py`: Training script for YOLOv8 model.
- `train_deeplab.py`: Training script for DeepLabV3 model.
- `pallet_detection_node.py`: ROS2 inference node for pallet detection and ground segmentation.
- `publisher_node.py`: ROS2 node to publish test images (for simulation).
- `data.yaml`: YOLO dataset configuration.
- `augment.py`: Data augmentation script for training.
- `evaluate_yolo.py`: Evaluates YOLOv8 mAP.
- `evaluate_deeplab.py`: Evaluates DeepLabV3 IoU.
- `convert_txt_to_png.py`: Converts YOLO .txt labels to .png masks for DeepLabV3.

## Models
- YOLOv8: [Google Drive link: https://drive.google.com/file/d/1bOc-T8NBDe59igy3AM9X2z7fv4XlLphO/view?usp=sharing]
- DeepLabV3: [Google Drive link: https://drive.google.com/file/d/1zj5LB2-N_I2cezmUr53mLT19s4DK2Q3f/view?usp=sharing]

## Prerequisites
- Ubuntu 22.04
- ROS2 Humble: https://docs.ros.org/en/humble/Installation.html
- NVIDIA AGX Orin (for inference)
- ZED 2i camera with ROS2 package: https://www.stereolabs.com/docs/ros2/
- Python 3.10.12
- CUDA 11.7, cuDNN (recommended for inference)

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/damodar9062/pallet_detection_project.git
   cd pallet_detection_project