#!/usr/bin/env python3
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
        self.subscription
        self.get_logger().info('订阅者已启动，等待接收识别结果...')
    
    def listener_callback(self, msg):
        # 按照要求的格式打印：[接收端] 收到识别结果：{类别}，准确度：{置信度}
        if ':' in msg.data:
            try:
                class_name, confidence = msg.data.split(':')
                self.get_logger().info(f'[接收端] 收到识别结果：{class_name}，准确度：{confidence}')
            except:
                self.get_logger().info(f'[接收端] 收到消息：{msg.data}')
        else:
            self.get_logger().info(f'[接收端] 收到消息：{msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = FashionSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
