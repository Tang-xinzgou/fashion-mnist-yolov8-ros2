#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class DetectionPublisher(Node):
    def __init__(self):
        super().__init__('detection_publisher')
        self.publisher_ = self.create_publisher(String, '/detection_result', 10)
        self.timer = self.create_timer(2.0, self.timer_callback)
        self.results = [
            ("T-shirt/top", 0.9567),
            ("Dress", 0.8923),
            ("Ankle boot", 0.9345),
        ]
        self.index = 0
        self.get_logger().info('发布者启动')
    
    def timer_callback(self):
        if self.index < len(self.results):
            class_name, confidence = self.results[self.index]
            msg = String()
            msg.data = f'{class_name}:{confidence:.4f}'
            self.publisher_.publish(msg)
            self.get_logger().info(f'[发送端] 发布: {class_name}, 准确度: {confidence:.4f}')
            self.index += 1
        else:
            self.get_logger().info('所有结果已发送')
            self.index = 0
            time.sleep(3)

def main():
    rclpy.init()
    node = DetectionPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
