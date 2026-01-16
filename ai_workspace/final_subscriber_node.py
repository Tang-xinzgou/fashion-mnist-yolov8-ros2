#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class FinalSubscriber(Node):
    def __init__(self):
        super().__init__('final_subscriber')
        self.subscription = self.create_subscription(
            String,
            '/detection_result',
            self.listener_callback,
            10)
        
        print("\n" + "="*60)
        print("Fashion-MNIST ROS2 订阅者节点")
        print("="*60)
        print("状态: 已就绪")
        print("正在监听话题: /detection_result")
        print("等待接收识别结果...")
        print("="*60)
        print("")
    
    def listener_callback(self, msg):
        if ':' in msg.data:
            try:
                class_name, confidence = msg.data.split(':')
                print(f'[接收端] 收到识别结果：{class_name}，准确度：{confidence}')
            except:
                print(f'[接收端] 收到消息：{msg.data}')
        else:
            print(f'[接收端] 收到消息：{msg.data}')

def main():
    rclpy.init()
    node = FinalSubscriber()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("✅ 订阅者节点被用户中断")
        print("✅ 请截图此终端输出")
        print("="*60)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()