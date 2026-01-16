#!/usr/bin/env python3
"""
检查模型训练状态和性能
"""
import os
import torch
import yaml
import numpy as np
from pathlib import Path

print("="*80)
print("模型训练状态诊断")
print("="*80)

# 1. 检查模型文件
model_path = 'runs/detect/train/weights/best.pt'
print(f"1. 模型文件检查:")
print(f"   路径: {model_path}")
print(f"   存在: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    file_size = os.path.getsize(model_path) / (1024*1024)  # MB
    print(f"   大小: {file_size:.2f} MB")
    
    # 检查权重文件
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"   ✅ 权重文件可加载")
        
        # 检查关键信息
        keys = list(checkpoint.keys())
        print(f"   检查点键: {keys}")
        
        if 'model' in checkpoint:
            print(f"   ✅ 包含模型参数")
        if 'optimizer' in checkpoint:
            print(f"   ✅ 包含优化器状态")
        if 'epoch' in checkpoint:
            print(f"   ✅ 训练轮数: {checkpoint['epoch']}")
            
    except Exception as e:
        print(f"   ❌ 权重文件损坏: {e}")

# 2. 检查训练配置
print(f"\n2. 训练配置检查:")
config_path = 'fashion_full.yaml'
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"   ✅ 配置文件存在")
    print(f"     类别数: {config.get('nc', 'N/A')}")
    print(f"     类别名: {config.get('names', 'N/A')}")
    print(f"     训练集: {config.get('train', 'N/A')}")
    print(f"     验证集: {config.get('val', 'N/A')}")
else:
    print(f"   ❌ 配置文件不存在")

# 3. 检查训练结果
print(f"\n3. 训练结果检查:")
train_dir = 'runs/detect/train'
if os.path.exists(train_dir):
    # 检查args.yaml
    args_path = os.path.join(train_dir, 'args.yaml')
    if os.path.exists(args_path):
        with open(args_path, 'r') as f:
            args = yaml.safe_load(f)
        print(f"   ✅ 训练参数:")
        print(f"     模型: {args.get('model', 'N/A')}")
        print(f"     轮数: {args.get('epochs', 'N/A')}")
        print(f"     批次大小: {args.get('batch', 'N/A')}")
        print(f"     图片尺寸: {args.get('imgsz', 'N/A')}")
    
    # 检查训练结果图表
    results_files = ['results.png', 'results.csv', 'confusion_matrix.png']
    for file in results_files:
        file_path = os.path.join(train_dir, file)
        if os.path.exists(file_path):
            print(f"   ✅ {file} 存在")
        else:
            print(f"   ❌ {file} 不存在")

# 4. 检查数据集
print(f"\n4. 数据集检查:")
val_dir = 'fashion_mnist_yolo/images/val'
if os.path.exists(val_dir):
    val_images = list(Path(val_dir).glob('*.png'))[:5]
    print(f"   验证集图片数量: {len(list(Path(val_dir).glob('*.png')))}")
    print(f"   前5张图片: {[img.name for img in val_images]}")
else:
    print(f"   ❌ 验证集目录不存在")

# 5. 模型推理测试
print(f"\n5. 模型推理测试:")
if os.path.exists(model_path) and os.path.exists(val_dir):
    try:
        from ultralytics import YOLO
        
        # 加载模型
        model = YOLO(model_path)
        print(f"   ✅ 模型加载成功")
        
        # 测试第一张图片
        test_img = f'fashion_mnist_yolo/images/val/val_00000.png'
        if os.path.exists(test_img):
            print(f"   测试图片: {test_img}")
            
            # 尝试不同的置信度阈值
            for conf in [0.01, 0.05, 0.1, 0.25, 0.5]:
                print(f"\n   置信度阈值 {conf}:")
                try:
                    results = model.predict(test_img, conf=conf, verbose=False)
                    
                    if results and results[0].boxes is not None:
                        num_boxes = len(results[0].boxes)
                        if num_boxes > 0:
                            box = results[0].boxes[0]
                            cls_id = int(box.cls[0]) if box.cls.numel() > 0 else -1
                            conf_score = float(box.conf[0]) if box.conf.numel() > 0 else 0
                            print(f"      检测到 {num_boxes} 个目标")
                            print(f"      第一个目标: 类别={cls_id}, 置信度={conf_score:.4f}")
                        else:
                            print(f"      有框对象但长度为0")
                    else:
                        print(f"      未检测到目标")
                except Exception as e:
                    print(f"      推理错误: {e}")
        else:
            print(f"   测试图片不存在")
            
    except Exception as e:
        print(f"   ❌ 模型加载/推理失败: {e}")

print("\n" + "="*80)
print("诊断完成")
print("="*80)
