import sys
print("Python 版本:", sys.version)
print("Python 路径:", sys.executable)

try:
    import rclpy
    rclpy.init()
    print("✅ ROS2 初始化成功")
    rclpy.shutdown()
except Exception as e:
    print(f"❌ ROS2 错误: {e}")

try:
    import ultralytics
    print(f"✅ Ultralytics 版本: {ultralytics.__version__}")
except Exception as e:
    print(f"❌ Ultralytics 错误: {e}")

try:
    import torch
    print(f"✅ PyTorch 版本: {torch.__version__}")
except Exception as e:
    print(f"❌ PyTorch 错误: {e}")
