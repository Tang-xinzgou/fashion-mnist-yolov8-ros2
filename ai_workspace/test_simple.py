#!/usr/bin/env python3
print("脚本开始执行...")

# 检查必要包
try:
    import pandas
    print("✓ pandas 已安装")
except ImportError:
    print("✗ pandas 未安装")

try:
    from PIL import Image
    print("✓ PIL/Pillow 已安装")
except ImportError:
    print("✗ PIL/Pillow 未安装")

# 检查数据文件
import os
if os.path.exists("archive(1)/fashion-mnist_train.csv"):
    print("✓ 找到训练数据文件")
else:
    print("✗ 未找到训练数据文件")

print("脚本执行完成")
