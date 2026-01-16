import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
import os

print("=== 直接训练 Fashion-MNIST ===")

# 1. 加载数据
train_df = pd.read_csv('archive(1)/fashion-mnist_train.csv')
test_df = pd.read_csv('archive(1)/fashion-mnist_test.csv')

print(f"训练集大小: {len(train_df)} 张图片")
print(f"测试集大小: {len(test_df)} 张图片")

# 2. 创建临时数据文件夹（用于YOLO训练）
def create_temp_dataset(df, split='train'):
    img_dir = f'temp_fashion_mnist/images/{split}'
    label_dir = f'temp_fashion_mnist/labels/{split}'
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    print(f"处理{split}集...")
    for i in range(min(100, len(df))):  # 先处理100张测试
        label = df.iloc[i, 0]
        pixels = df.iloc[i, 1:].values.reshape(28, 28).astype(np.uint8)
        
        # 保存为64x64图片
        from PIL import Image
        canvas = Image.new('L', (64, 64), 0)
        small_img = Image.fromarray(pixels, mode='L')
        canvas.paste(small_img, ((64-28)//2, (64-28)//2))
        canvas.save(f'{img_dir}/{split}_{i:05d}.png')
        
        # 创建标签文件
        with open(f'{label_dir}/{split}_{i:05d}.txt', 'w') as f:
            f.write(f'{label} 0.5 0.5 0.4375 0.4375\n')
    
    return len(df)

# 创建训练和验证数据
train_count = create_temp_dataset(train_df, 'train')
val_count = create_temp_dataset(test_df, 'val')

# 3. 创建数据集配置文件
config = {
    'path': os.path.abspath('temp_fashion_mnist'),
    'train': 'images/train',
    'val': 'images/val',
    'nc': 10,
    'names': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
}

import yaml
with open('temp_fashion.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"\n数据集准备完成!")
print(f"配置文件: temp_fashion.yaml")
print(f"开始训练YOLOv8模型...\n")

# 4. 开始训练
model = YOLO('yolov8n.pt')
results = model.train(
    data='temp_fashion.yaml',
    epochs=5,
    imgsz=64,
    batch=16,
    verbose=True
)

print("训练完成!")
