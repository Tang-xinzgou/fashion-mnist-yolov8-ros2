import sys
print("="*60)
print("环境验证结果")
print("="*60)
print(f"Python 版本: {sys.version}")
print(f"Python 路径: {sys.executable}")

# 测试ROS2
try:
    import rclpy
    rclpy.init()
    print("✅ ROS2 初始化成功")
    rclpy.shutdown()
except Exception as e:
    print(f"❌ ROS2 错误: {e}")

# 测试Ultralytics
try:
    import ultralytics
    print(f"✅ Ultralytics 版本: {ultralytics.__version__}")
except Exception as e:
    print(f"❌ Ultralytics 错误: {e}")

# 测试PyTorch
try:
    import torch
    print(f"✅ PyTorch 版本: {torch.__version__}")
    print(f"✅ CUDA 可用: {torch.cuda.is_available()}")
except Exception as e:
    print(f"❌ PyTorch 错误: {e}")

print("="*60)
