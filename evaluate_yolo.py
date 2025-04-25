from ultralytics import YOLO
# Load trained YOLOv8 model
model = YOLO('/home/dhamodarlinux/ros2_ws/data/pallets/runs/detect/train/weights/best.pt')

# Evaluate on test set
metrics = model.val(data='/home/dhamodarlinux/ros2_ws/data/pallets/pallet_detection/data.yaml', split='test')
print(f"Test mAP: {metrics.box.map}")

# Save to evaluation.txt
with open('/home/dhamodarlinux/ros2_ws/data/pallets/evaluation.txt', 'w') as f:
    f.write(f"Pallet Detection Test mAP: {metrics.box.map}\n")