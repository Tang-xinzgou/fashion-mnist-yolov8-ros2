#!/usr/bin/env python3
"""
独立的ROS2发布者节点
在conda环境中运行，避免版本冲突
"""

import sys
import os
import threading
import time

# 添加必要的路径
sys.path.append('/opt/ros/humble/lib/python3.10/site-packages')

# 导入ROS2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# 导入YOLO
from ultralytics import YOLO

class FashionPublisher(Node):
    def __init__(self):
        super().__init__('fashion_publisher')
        self.publisher_ = self.create_publisher(String, '/detection_result', 10)
        
        # 加载训练好的模型
        model_path = 'runs/detect/train8/weights/best.pt'
        print(f"加载模型: {model_path}")
        
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            print("✅ 模型加载成功")
        else:
            print("❌ 模型文件不存在，使用模拟数据")
            self.model = None
        
        # 类别名称
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        # 测试图片路径
        test_dir = 'fashion_mnist_yolo/images/val/'
        if os.path.exists(test_dir):
            self.test_images = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.png')])[:3]
        else:
            self.test_images = []
            print("❌ 测试图片目录不存在")
        
        self.current_image_index = 0
        self.get_logger().info('Fashion-MNIST 发布者节点已启动')
        
        # 开始发布
        self.timer = self.create_timer(3.0, self.publish_detection)
    
    def publish_detection(self):
        msg = String()
        
        if self.model and self.test_images:
            # 使用真实模型预测
            img_path = self.test_images[self.current_image_index]
            try:
                results = self.model.predict(img_path, conf=0.25)
                
                if results[0].boxes:
                    box = results[0].boxes[0]
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    class_name = self.class_names[cls_id]
                    result_str = f"{class_name}:{conf:.4f}"
                    msg.data = result_str
                    self.get_logger().info(f'发布: "{result_str}"')
                else:
                    msg.data = "未检测到目标:0.0000"
            except Exception as e:
                msg.data = f"检测失败:{str(e)[:30]}"
        else:
            # 使用模拟数据
            sim_results = [
                ("T-shirt/top", 0.9567),
                ("Dress", 0.8923),
                ("Ankle boot", 0.9345),
            ]
            class_name, confidence = sim_results[self.current_image_index % len(sim_results)]
            result_str = f"{class_name}:{confidence:.4f}"
            msg.data = result_str
            self.get_logger().info(f'发布(模拟): "{result_str}"')
        
        # 发布消息
        self.publisher_.publish(msg)
        
        # 更新索引
        self.current_image_index += 1
        if self.test_images and self.current_image_index >= len(self.test_images):
            self.current_image_index = 0

def main():
    print("="*60)
    print("Fashion-MNIST ROS2 发布者节点")
    print("="*60)
    
    # 初始化ROS2
    rclpy.init()
    
    # 创建节点
    node = FashionPublisher()
    
    try:
        # 运行节点
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n节点被中断")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
