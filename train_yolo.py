#!/usr/bin/env python3
import os
import torch   
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
model.train(
    data='/home/dhamodarlinux/ros2_ws/data/pallets/pallet_detection/data.yaml',
    epochs=100,  # Increased epochs
    patience=0,
    cache=True,
    batch=8,
    imgsz=640,
    device=0 if torch.cuda.is_available() else 'cpu',
    lr0=0.001,
    augment=True,
    workers=2,
    project='/home/dhamodarlinux/ros2_ws/data/pallets/runs/detect',
    name='train',
    exist_ok=True,

)
model.export(format='onnx', simplify=True)





