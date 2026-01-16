import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TestSubscriber(Node):
    def __init__(self):
        super().__init__('test_subscriber')
        self.subscription = self.create_subscription(
            String,
            '/detection_result',
            self.listener_callback,
            10)
        print("订阅者已启动，等待接收消息...")
    
    def listener_callback(self, msg):
        if ':' in msg.data:
            class_name, confidence = msg.data.split(':')
            print(f'[接收端] 收到识别结果：{class_name}，准确度：{confidence}')
        else:
            print(f'[接收端] 收到消息：{msg.data}')

def main():
    rclpy.init()
    node = TestSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
