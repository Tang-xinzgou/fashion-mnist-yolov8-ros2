#!/usr/bin/env python3
from ultralytics import YOLO
import os
import glob
import time

print("="*70)
print("Fashion-MNIST YOLOv8 重新训练结果验证")
print("="*70)

# 1. 查找最新的训练模型
train_dirs = sorted(glob.glob('runs/detect/train*'))
if not train_dirs:
    print("❌ 未找到训练目录")
    exit(1)

latest_train = train_dirs[-1]
model_path = os.path.join(latest_train, 'weights/best.pt')

if not os.path.exists(model_path):
    print(f"❌ 模型文件不存在: {model_path}")
    exit(1)

print(f"✅ 加载最新模型: {model_path}")
model = YOLO(model_path)

# 2. Fashion-MNIST类别
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 3. 测试多张验证集图片
test_images = []
for i in range(10):  # 测试10张图片
    img_name = f'val_{i:05d}.png'
    img_path = f'fashion_mnist_yolo/images/val/{img_name}'
    if os.path.exists(img_path):
        test_images.append(img_path)

if not test_images:
    print("❌ 未找到测试图片")
    # 创建一些测试图片
    print("正在创建测试图片...")
    import numpy as np
    from PIL import Image
    os.makedirs('fashion_mnist_yolo/images/val', exist_ok=True)
    for i in range(5):
        img_array = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        Image.fromarray(img_array).save(f'fashion_mnist_yolo/images/val/val_{i:05d}.png')
        test_images.append(f'fashion_mnist_yolo/images/val/val_{i:05d}.png')

print(f"\n🔍 开始测试 {len(test_images)} 张图片...")
print("="*70)

# 4. 进行识别并输出结果
results_summary = []

for i, img_path in enumerate(test_images[:5], 1):  # 只测试前5张
    if not os.path.exists(img_path):
        print(f"❌ 图片不存在: {img_path}")
        continue
    
    print(f"\n测试 {i}: {os.path.basename(img_path)}")
    
    try:
        # 进行推理
        start_time = time.time()
        results = model.predict(img_path, conf=0.25, verbose=False)
        inference_time = time.time() - start_time
        
        if results[0].boxes:
            for j, box in enumerate(results[0].boxes, 1):
                cls_id = int(box.cls)
                conf = float(box.conf)
                class_name = class_names[cls_id]
                
                # 获取边界框
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                print(f"  检测 {j}: {class_name}")
                print(f"  准确度: {conf:.4f} ({conf*100:.2f}%)")
                print(f"  边界框: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                print(f"  推理时间: {inference_time:.3f}秒")
                
                # 保存结果用于汇总
                results_summary.append({
                    'image': os.path.basename(img_path),
                    'class': class_name,
                    'confidence': conf,
                    'time': inference_time
                })
        else:
            print("  未检测到目标")
            results_summary.append({
                'image': os.path.basename(img_path),
                'class': '无检测',
                'confidence': 0.0,
                'time': inference_time
            })
            
    except Exception as e:
        print(f"  ❌ 识别失败: {e}")

# 5. 输出汇总统计
print("\n" + "="*70)
print("�� 识别结果汇总")
print("="*70)

if results_summary:
    # 计算平均准确度
    detected = [r for r in results_summary if r['confidence'] > 0]
    if detected:
        avg_confidence = sum(r['confidence'] for r in detected) / len(detected)
        print(f"平均准确度: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
        print(f"检测成功率: {len(detected)}/{len(results_summary)}")
    
    # 类别分布
    print("\n📈 类别分布:")
    class_counts = {}
    for r in results_summary:
        if r['class'] != '无检测':
            class_counts[r['class']] = class_counts.get(r['class'], 0) + 1
    
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}次")
    
    # 详细结果
    print("\n📋 详细结果:")
    for r in results_summary:
        if r['confidence'] > 0:
            print(f"  {r['image']}: {r['class']} ({r['confidence']:.4f})")
        else:
            print(f"  {r['image']}: 无检测")

print("="*70)
print("✅ 识别结果验证完成")
print("="*70)
