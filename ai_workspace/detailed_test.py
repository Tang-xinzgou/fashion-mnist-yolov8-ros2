#!/usr/bin/env python3
from ultralytics import YOLO
import os
import glob

print("="*70)
print("Fashion-MNIST YOLOv8 训练结果详细报告")
print("="*70)

# 1. 查找所有训练结果
train_dirs = glob.glob('runs/detect/train*')
print(f"找到 {len(train_dirs)} 个训练目录:")
for d in sorted(train_dirs):
    print(f"  - {d}")

# 2. 加载最新训练的模型
if not train_dirs:
    print("❌ 未找到训练目录")
    exit(1)

latest_train = max(train_dirs, key=os.path.getmtime)
model_path = os.path.join(latest_train, 'weights/best.pt')

if os.path.exists(model_path):
    print(f"\n✅ 加载最新模型: {model_path}")
    model = YOLO(model_path)
    
    # 3. 模型信息
    print("\n模型信息:")
    print(f"  类别数: {model.model.nc}")
    print(f"  输入尺寸: {model.model.args.get('imgsz', '未知')}")
    
    # 4. 测试识别
    print("\n" + "-"*70)
    print("识别测试结果:")
    print("-"*70)
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # 测试几张图片
    for i in range(5):
        img_name = f'val_{i:05d}.png'
        img_path = f'fashion_mnist_yolo/images/val/{img_name}'
        
        if os.path.exists(img_path):
            results = model.predict(img_path, conf=0.25, verbose=False)
            
            if results[0].boxes:
                box = results[0].boxes[0]
                cls_id = int(box.cls)
                conf = float(box.conf)
                class_name = class_names[cls_id]
                
                print(f"图片: {img_name}")
                print(f"  类别: {class_name} (ID: {cls_id})")
                print(f"  置信度: {conf:.4f}")
                print(f"  准确度百分比: {conf*100:.2f}%")
                print()
            else:
                print(f"图片: {img_name} - 未检测到目标\n")
        else:
            print(f"图片不存在: {img_path}\n")
    
    # 5. 验证集评估
    print("\n" + "-"*70)
    print("验证集评估:")
    print("-"*70)
    
    try:
        # 在验证集上评估
        metrics = model.val(data='fashion_mnist_yolo', split='val')
        print(f"mAP50: {metrics.box.map:.4f}")
        print(f"mAP50-95: {metrics.box.map50:.4f}")
        print(f"精确度: {metrics.box.p:.4f}")
        print(f"召回率: {metrics.box.r:.4f}")
    except Exception as e:
        print(f"评估失败: {e}")
        print("（可能需要fashion_mnist_yolo数据集配置文件）")
    
else:
    print(f"❌ 模型文件不存在: {model_path}")

print("\n" + "="*70)
print("报告完成")
print("="*70)
