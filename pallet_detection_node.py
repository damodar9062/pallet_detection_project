#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from torchvision.models.segmentation import deeplabv3_resnet50
import albumentations as A
import onnxruntime as ort

class PalletDetectionNode(Node):
    def __init__(self):
        super().__init__('pallet_detection_node')
        self.bridge = CvBridge()

        # QoS profile
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # YOLOv8 ONNX model
        try:
            self.yolo_session = ort.InferenceSession('/home/dhamodarlinux/ros2_ws/data/pallets/runs/detect/train/weights/best.onnx')
            self.input_name = self.yolo_session.get_inputs()[0].name
            self.get_logger().info('YOLOv8 model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLOv8 model: {str(e)}')
            raise

        # DeepLabV3 model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.deeplab = deeplabv3_resnet50(weights=None, num_classes=2, aux_loss=False)
            checkpoint = torch.load(
                '/home/dhamodarlinux/ros2_ws/data/pallets/deeplabv3.pth',
                map_location='cpu',
                weights_only=True
            )
            self.deeplab.load_state_dict(checkpoint, strict=False)
            self.deeplab.to(self.device)
            self.deeplab.eval()
            self.get_logger().info('DeepLabV3 model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load DeepLabV3 model: {str(e)}')
            raise

        # Transformations for DeepLabV3
        self.transform = A.Compose([
            A.Resize(520, 520),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # Subscriptions and publishers
        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, qos)
        self.pallet_pub = self.create_publisher(Image, '/pallet_detection', qos)
        self.ground_pub = self.create_publisher(Image, '/ground_segmentation', qos)

    def preprocess_yolo(self, image):
        img = cv2.resize(image, (640, 640))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

    def postprocess_yolo(self, outputs, cv_image, original_shape):
        self.get_logger().info('Processing YOLOv8 output')
        try:
            boxes = outputs[0][0]  # Shape: [5, 8400]
            img_h, img_w = original_shape[:2]
            scale = min(640 / img_w, 640 / img_h)
            offset_x = (640 - img_w * scale) / 2
            offset_y = (640 - img_h * scale) / 2

            result_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB).copy()
            for i in range(boxes.shape[1]):
                conf = boxes[4, i]
                if conf > 0.5:  # Confidence threshold
                    x, y, w, h = boxes[:4, i]
                    x = (x - offset_x) / scale
                    y = (y - offset_y) / scale
                    w = w / scale
                    h = h / scale
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(result_img, f'Pallet {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.get_logger().error(f'YOLOv8 postprocessing error: {str(e)}')
            raise

    def image_callback(self, msg):
        try:
            self.get_logger().info('Received image message')
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.get_logger().info(f'Image shape: {cv_image.shape}')

            # YOLOv8 inference
            self.get_logger().info('Running YOLOv8 inference')
            yolo_input = self.preprocess_yolo(cv_image)
            yolo_outputs = self.yolo_session.run(None, {self.input_name: yolo_input})
            pallet_img = self.postprocess_yolo(yolo_outputs, cv_image, cv_image.shape)

            # DeepLabV3 inference
            self.get_logger().info('Running DeepLabV3 inference')
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            transformed = self.transform(image=rgb_image)
            input_tensor = torch.tensor(transformed['image'], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.deeplab(input_tensor)['out']
            ground_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            ground_mask = cv2.resize(ground_mask.astype(np.uint8), (cv_image.shape[1], cv_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            ground_img = cv_image.copy()
            ground_img[ground_mask == 1] = [0, 255, 0]  # Green for ground

            # Publish results
            self.get_logger().info('Publishing results')
            self.pallet_pub.publish(self.bridge.cv2_to_imgmsg(pallet_img, 'bgr8'))
            self.ground_pub.publish(self.bridge.cv2_to_imgmsg(ground_img, 'bgr8'))
            self.get_logger().info('Published pallet and ground images')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

def main():
    rclpy.init()
    node = PalletDetectionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()