#!/usr/bin/env python3
"""
Fashion-MNIST ROS2 发布者节点
发布识别结果（类别和准确度）
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
import sys
import os

# 添加项目路径
sys.path.append('/home/txz/fashion_mnist_ros_peoject/ai_workspace')

class DetectionPublisher(Node):
    def __init__(self):
        super().__init__('detection_publisher')
        
        # 创建发布者
        self.publisher_ = self.create_publisher(String, '/detection_result', 10)
        
        # 定时器，每2秒发布一次
        self.timer = self.create_timer(2.0, self.timer_callback)
        
        # 加载识别结果
        self.detection_results = self.load_detection_results()
        
        self.result_index = 0
        self.get_logger().info('✅ Fashion-MNIST 识别结果发布者已启动')
        self.get_logger().info('   等待订阅者连接...')
    
    def load_detection_results(self):
        """加载或模拟识别结果"""
        # 这里可以替换为实际模型的识别结果
        results = [
            ("图片: val_00000.png", "T-shirt/top", 0.9567),
            ("图片: val_00001.png", "Dress", 0.8923),
            ("图片: val_00002.png", "Ankle boot", 0.9345),
            ("图片: val_00003.png", "Pullover", 0.8765),
            ("图片: val_00004.png", "Sneaker", 0.9034),
        ]
        return results
    
    def timer_callback(self):
        """定时发布识别结果"""
        if self.result_index < len(self.detection_results):
            img_name, class_name, confidence = self.detection_results[self.result_index]
            
            # 创建消息
            msg = String()
            msg.data = f'{class_name}:{confidence:.4f}'
            
            # 发布消息
            self.publisher_.publish(msg)
            
            # 打印日志
            self.get_logger().info(f'[发送端] 发布识别结果: {class_name}, 准确度: {confidence:.4f}')
            
            self.result_index += 1
            
            # 如果所有结果都已发送，等待一会儿后重启
            if self.result_index >= len(self.detection_results):
                self.get_logger().info('✅ 所有识别结果已发送完成')
                self.result_index = 0
                time.sleep(5)  # 等待5秒后重新开始

def main(args=None):
    rclpy.init(args=args)
    node = DetectionPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
