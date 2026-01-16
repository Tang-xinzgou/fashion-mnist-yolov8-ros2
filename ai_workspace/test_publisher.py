import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class TestPublisher(Node):
    def __init__(self):
        super().__init__('test_publisher')
        self.publisher_ = self.create_publisher(String, '/detection_result', 10)
        
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        self.sim_results = [
            ("T-shirt/top", 0.9567),
            ("Dress", 0.8923),
            ("Ankle boot", 0.9345),
        ]
        
        self.index = 0
        self.counter = 0
        
        print("发布者启动，开始发布消息...")
        self.timer = self.create_timer(2.0, self.publish_detection)
    
    def publish_detection(self):
        self.counter += 1
        if self.counter > 5:
            print("已发布5条消息，停止")
            rclpy.shutdown()
            return
        
        class_name, confidence = self.sim_results[self.index]
        result_str = f"{class_name}:{confidence:.4f}"
        
        msg = String()
        msg.data = result_str
        self.publisher_.publish(msg)
        
        print(f"发布: {result_str}")
        self.index = (self.index + 1) % len(self.sim_results)

def main():
    rclpy.init()
    node = TestPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
