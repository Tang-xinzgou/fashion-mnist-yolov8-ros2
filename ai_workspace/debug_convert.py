import pandas as pd
import numpy as np
import os
print("=== 调试模式 ===")

# 1. 检查文件
print("1. 检查数据文件...")
train_path = "archive(1)/fashion-mnist_train.csv"
test_path = "archive(1)/fashion-mnist_test.csv"
print(f"训练文件存在: {os.path.exists(train_path)}")
print(f"测试文件存在: {os.path.exists(test_path)}")

# 2. 尝试读取少量数据
print("\n2. 尝试读取数据...")
try:
    df_train = pd.read_csv(train_path, nrows=10)
    df_test = pd.read_csv(test_path, nrows=10)
    print(f"训练数据形状: {df_train.shape}")
    print(f"测试数据形状: {df_test.shape}")
    print(f"前5个标签: {df_train.iloc[:5, 0].tolist()}")
    print("✓ 数据读取成功")
except Exception as e:
    print(f"✗ 数据读取失败: {e}")

# 3. 检查输出目录
print("\n3. 创建输出目录...")
os.makedirs("debug_output/images", exist_ok=True)
os.makedirs("debug_output/labels", exist_ok=True)
print("✓ 目录创建完成")

print("\n=== 调试完成 ===")
