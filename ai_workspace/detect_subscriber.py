#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class DetectionSubscriber(Node):
    def __init__(self):
        super().__init__('detection_subscriber')
        self.subscription = self.create_subscription(
            String,
            '/detection_result',
            self.callback,
            10)
        self.get_logger().info('订阅者启动，等待接收...')
    
    def callback(self, msg):
        if ':' in msg.data:
            class_name, confidence = msg.data.split(':')
            self.get_logger().info(f'[接收端] 收到识别结果：{class_name}，准确度：{confidence}')
        else:
            self.get_logger().info(f'[接收端] 收到：{msg.data}')

def main():
    rclpy.init()
    node = DetectionSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
