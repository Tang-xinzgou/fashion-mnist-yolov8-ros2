import pandas as pd
import os

print("正在检查数据文件...")
data_path = "archive(1)/fashion-mnist_train.csv"
if os.path.exists(data_path):
    print(f"✓ 找到数据文件: {data_path}")
    # 尝试读取前5行
    df = pd.read_csv(data_path, nrows=5)
    print(f"✓ 成功读取数据，形状: {df.shape}")
    print("前5行标签:", df.iloc[:5, 0].tolist())
else:
    print(f"✗ 数据文件不存在: {data_path}")
    print("当前目录内容:", os.listdir("."))
    print("archive(1)目录内容:", os.listdir("archive(1)") if os.path.exists("archive(1)") else "archive(1)不存在")
