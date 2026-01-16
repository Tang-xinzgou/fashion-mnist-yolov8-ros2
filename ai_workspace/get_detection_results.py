#!/usr/bin/env python3
"""
Fashion-MNIST 识别结果输出脚本
"""
import numpy as np
from ultralytics import YOLO
import os
import glob
import sys

print("="*60)
print("Fashion-MNIST 识别结果输出")
print("="*60)

# 检查必要的库
print(f"NumPy版本: {np.__version__}")

# 查找最新的训练模型
train_dirs = sorted(glob.glob('runs/detect/train*'))
if not train_dirs:
    print("❌ 未找到训练目录")
    sys.exit(1)

model_path = os.path.join(train_dirs[-1], 'weights/best.pt')
if not os.path.exists(model_path):
    print(f"❌ 模型文件不存在: {model_path}")
    sys.exit(1)

print(f"✅ 加载模型: {os.path.basename(train_dirs[-1])}")
print(f"📁 模型路径: {model_path}")

# 加载模型
try:
    model = YOLO(model_path)
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit(1)

# 类别名称
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 测试图片
test_images = []
for i in range(5):  # 测试5张图片
    img_name = f'val_{i:05d}.png'
    img_path = f'fashion_mnist_yolo/images/val/{img_name}'
    if os.path.exists(img_path):
        test_images.append(img_path)

if not test_images:
    print("❌ 未找到测试图片")
    sys.exit(1)

print(f"\n🔍 将测试 {len(test_images)} 张图片")
print("="*70)
print("\n📊 识别结果输出:")
print("="*70)

# 进行识别并输出结果
for i, img_path in enumerate(test_images, 1):
    img_name = os.path.basename(img_path)
    print(f"\n测试 {i}: {img_name}")
    print("-" * 40)
    
    try:
        # 进行推理
        results = model.predict(img_path, conf=0.25, verbose=False)
        
        if results[0].boxes:
            box = results[0].boxes[0]
            cls_id = int(box.cls)
            conf = float(box.conf)
            class_name = class_names[cls_id] if cls_id < len(class_names) else f"未知({cls_id})"
            
            print(f"  识别类别: {class_name}")
            print(f"  准确度: {conf:.4f}")
            print(f"  置信度: {conf*100:.2f}%")
        else:
            print("  未检测到目标")
    except Exception as e:
        print(f"  ❌ 识别失败: {e}")

print("\n" + "="*70)
print("✅ 识别结果输出完成")
print("📸 请截图此输出作为作业提交")
print("="*70)