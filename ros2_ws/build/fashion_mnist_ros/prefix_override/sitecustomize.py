import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/txz/fashion_mnist_ros_peoject/ros2_ws/install/fashion_mnist_ros'
