#!/usr/bin/env python3
"""
独立的ROS2订阅者节点
在conda环境中运行
"""

import sys
sys.path.append('/opt/ros/humble/lib/python3.10/site-packages')

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class FashionSubscriber(Node):
    def __init__(self):
        super().__init__('fashion_subscriber')
        self.subscription = self.create_subscription(
            String,
            '/detection_result',
            self.listener_callback,
            10)
        self.get_logger().info('订阅者节点已启动，等待接收识别结果...')
    
    def listener_callback(self, msg):
        # 按照作业要求格式打印
        if ':' in msg.data:
            try:
                class_name, confidence = msg.data.split(':')
                self.get_logger().info(f'[接收端] 收到识别结果：{class_name}，准确度：{confidence}')
            except:
                self.get_logger().info(f'[接收端] 收到消息：{msg.data}')
        else:
            self.get_logger().info(f'[接收端] 收到消息：{msg.data}')

def main():
    print("="*60)
    print("Fashion-MNIST ROS2 订阅者节点")
    print("="*60)
    
    rclpy.init()
    node = FashionSubscriber()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n节点被中断")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
