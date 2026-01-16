#!/usr/bin/env python3
import os, cv2, numpy as np, yaml
from ultralytics import YOLO
from datetime import datetime

print("Fashion-MNIST 训练")
print("="*60)

# 1. 生成数据
os.makedirs('fashion_data/images/train', exist_ok=True)
os.makedirs('fashion_data/labels/train', exist_ok=True)
os.makedirs('fashion_data/images/val', exist_ok=True)
os.makedirs('fashion_data/labels/val', exist_ok=True)

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 生成图片和标签
for cls_id in range(10):
    for i in range(50):  # 每类50张训练
        img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(f'fashion_data/images/train/train_{cls_id}_{i:03d}.png', img)
        with open(f'fashion_data/labels/train/train_{cls_id}_{i:03d}.txt', 'w') as f:
            f.write(f'{cls_id} 0.5 0.5 0.8 0.8')
    
    for i in range(10):  # 每类10张验证
        img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(f'fashion_data/images/val/val_{cls_id}_{i:03d}.png', img)
        with open(f'fashion_data/labels/val/val_{cls_id}_{i:03d}.txt', 'w') as f:
            f.write(f'{cls_id} 0.5 0.5 0.8 0.8')

# 2. 创建配置
config = {
    'path': os.path.abspath('.'),
    'train': 'fashion_data/images/train',
    'val': 'fashion_data/images/val',
    'nc': 10,
    'names': classes
}
with open('fashion_config.yaml', 'w') as f:
    yaml.dump(config, f)

# 3. 训练模型
print("开始训练...")
model = YOLO('yolov8n.pt')
results = model.train(
    data='fashion_config.yaml',
    epochs=20,
    imgsz=64,
    batch=16,
    device='cpu',
    workers=2,
    project='fashion_result',
    name='train',
    exist_ok=True
)

# 4. 测试
model_path = 'fashion_result/train/weights/best.pt'
if os.path.exists(model_path):
    trained = YOLO(model_path)
    
    # 测试3张
    for i in range(3):
        img = f'fashion_data/images/val/val_0_{i:03d}.png'
        if os.path.exists(img):
            print(f"\n测试图片 {i+1}: {os.path.basename(img)}")
            res = trained.predict(img, conf=0.25, verbose=False)
            
            if res[0].boxes and len(res[0].boxes) > 0:
                box = res[0].boxes[0]
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = classes[cls_id] if 0 <= cls_id < 10 else f"未知({cls_id})"
                print(f"  识别结果: {cls_name}")
                print(f"  准确度: {conf:.4f}")
            else:
                print("  未检测到目标")

print("\n✅ 训练完成")
