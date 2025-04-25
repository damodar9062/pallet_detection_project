#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImagePublisherNode(Node):
    def __init__(self):
        super().__init__('image_publisher_node')
        self.publisher = self.create_publisher(Image, '/image_raw', 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.image_path = '/home/dhamodarlinux/ros2_ws/data/pallets/pallet_detection/test/images/1564563279-4191911_jpg.rf.a41bbfd5341feaf147ac447bd26b2a74.jpg'
        self.get_logger().info(f'Publishing image from {self.image_path}')

    def timer_callback(self):
        try:
            img = cv2.imread(self.image_path)
            if img is None:
                self.get_logger().error(f'Failed to load image: {self.image_path}')
                return
            img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            self.publisher.publish(img_msg)
            self.get_logger().info('Published image')
        except Exception as e:
            self.get_logger().error(f'Error publishing image: {str(e)}')

def main():
    rclpy.init()
    node = ImagePublisherNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
