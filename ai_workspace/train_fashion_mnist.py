#!/usr/bin/env python3
"""
Fashion-MNIST YOLOv8 训练脚本
"""
from ultralytics import YOLO
import os
import yaml

def main():
    print("="*60)
    print("Fashion-MNIST YOLOv8 模型训练")
    print("="*60)
    
    # 1. 加载配置文件
    config_file = 'fashion_full.yaml'
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"数据集类别: {config.get('names', [])}")
    print(f"类别数量: {config.get('nc', 0)}")
    
    # 2. 加载YOLOv8模型
    print("加载YOLOv8n模型...")
    model = YOLO('yolov8n.pt')
    
    # 3. 训练参数
    train_args = {
        'data': config_file,
        'epochs': 50,
        'imgsz': 64,
        'batch': 16,
        'device': 'cpu',
        'workers': 4,
        'project': 'fashion_mnist_training',
        'name': 'train',
        'exist_ok': True,
        'verbose': True
    }
    
    print("训练参数:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    
    # 4. 开始训练
    print("\n开始训练...")
    print("-"*60)
    results = model.train(**train_args)
    
    print("\n" + "="*60)
    print("✅ 训练完成！")
    print("模型保存位置:")
    print("  runs/detect/train/weights/best.pt")
    print("="*60)

if __name__ == "__main__":
    main()
