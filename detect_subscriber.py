#!/usr/bin/env python3
"""
Fashion-MNIST ROS2 订阅者节点
接收并显示识别结果
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class DetectionSubscriber(Node):
    def __init__(self):
        super().__init__('detection_subscriber')
        
        # 创建订阅者
        self.subscription = self.create_subscription(
            String,
            '/detection_result',
            self.listener_callback,
            10
        )
        
        self.get_logger().info('✅ Fashion-MNIST 识别结果订阅者已启动')
        self.get_logger().info('   等待接收识别结果...')
    
    def listener_callback(self, msg):
        """接收消息的回调函数"""
        try:
            # 消息格式应为 "类别:准确度"
            if ':' in msg.data:
                class_name, confidence = msg.data.split(':')
                
                # 按照要求格式输出
                self.get_logger().info(f'[接收端] 收到识别结果：{class_name}，准确度：{confidence}')
            else:
                self.get_logger().info(f'[接收端] 收到消息：{msg.data}')
                
        except Exception as e:
            self.get_logger().info(f'[接收端] 消息解析错误：{e}')

def main(args=None):
    rclpy.init(args=args)
    node = DetectionSubscriber()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
