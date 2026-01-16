#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO
import os

class FashionPublisher(Node):
    def __init__(self):
        super().__init__('fashion_publisher')
        self.publisher_ = self.create_publisher(String, '/detection_result', 10)
        
        # 加载训练好的模型
        model_path = os.path.expanduser('~/fashion_mnist_ros_peoject/ai_workspace/runs/detect/train8/weights/best.pt')
        print(f"尝试加载模型: {model_path}")
        print(f"文件是否存在: {os.path.exists(model_path)}")
        
        self.model = YOLO(model_path)
        self.get_logger().info('YOLOv8模型加载成功')
        
        # 类别名称
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        # 测试图片路径
        self.test_image = os.path.expanduser('~/fashion_mnist_ros_peoject/ai_workspace/fashion_mnist_yolo/images/val/val_00000.png')
        
        timer_period = 3.0  # 每3秒发布一次
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('发布者节点已启动，开始发布识别结果...')
    
    def timer_callback(self):
        try:
            # 进行推理
            results = self.model(self.test_image)
            msg = String()
            
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
                self.get_logger().warn('未检测到目标')
            
            self.publisher_.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'推理失败: {e}')
            msg = String()
            msg.data = f"推理失败:{str(e)[:30]}"
            self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FashionPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('节点被关闭')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
