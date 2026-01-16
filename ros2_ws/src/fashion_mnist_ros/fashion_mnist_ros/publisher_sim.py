#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
import random

class FashionSimPublisher(Node):
    def __init__(self):
        super().__init__('fashion_sim_publisher')
        self.publisher_ = self.create_publisher(String, '/detection_result', 10)
        
        # 类别名称
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        # 模拟的识别结果（基于真实训练结果）
        self.sim_results = [
            ("T-shirt/top", 0.9567),
            ("Dress", 0.8923), 
            ("Ankle boot", 0.9345),
            ("Pullover", 0.9128),
            ("Sneaker", 0.9034)
        ]
        
        self.index = 0
        timer_period = 2.0
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('Fashion-MNIST 模拟发布者已启动...')
    
    def timer_callback(self):
        # 循环使用模拟结果
        class_name, confidence = self.sim_results[self.index]
        result_str = f"{class_name}:{confidence:.4f}"
        
        msg = String()
        msg.data = result_str
        self.publisher_.publish(msg)
        
        # 输出到终端
        self.get_logger().info(f'发布: "{result_str}"')
        
        # 更新索引
        self.index = (self.index + 1) % len(self.sim_results)

def main(args=None):
    rclpy.init(args=args)
    node = FashionSimPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('节点被关闭')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
