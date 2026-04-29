from mmdet.datasets import build_dataset
from mmcv import Config

cfg = Config.fromfile('你的配置文件路径.py')

# 打印完整的数据集配置，方便检查路径
print("=" * 50)
print("训练集配置：")
print(f"type: {cfg.data.train.type}")
print(f"ann_file: {cfg.data.train.ann_file}")
print(f"img_prefix: {cfg.data.train.img_prefix}")
if hasattr(cfg.data.train, 'data_root'):
    print(f"data_root: {cfg.data.train.data_root}")
print(f"pipeline 长度: {len(cfg.data.train.pipeline)}")

# 检查标注文件是否存在
import os
ann_file = cfg.data.train.ann_file
print(f"\n标注文件是否存在: {os.path.exists(ann_file)}")
if os.path.exists(ann_file):
    print(f"标注文件大小: {os.path.getsize(ann_file)} bytes")
    # 如果是 json 文件，读取前几个条目看看
    if ann_file.endswith('.json'):
        import json
        with open(ann_file, 'r') as f:
            data = json.load(f)
        print(f"JSON 顶层类型: {type(data)}")
        if isinstance(data, dict):
            print(f"JSON 顶层 keys: {list(data.keys())}")
            if 'images' in data:
                print(f"images 条目数: {len(data['images'])}")
                if len(data['images']) > 0:
                    print(f"第一张图片信息: {data['images'][0]}")
            if 'annotations' in data:
                print(f"annotations 条目数: {len(data['annotations'])}")
            if 'categories' in data:
                print(f"categories 条目数: {len(data['categories'])}")
                print(f"categories: {data['categories']}")
    elif ann_file.endswith('.pkl') or ann_file.endswith('.pickle'):
        import pickle
        with open(ann_file, 'rb') as f:
            data = pickle.load(f)
        print(f"pkl 数据类型: {type(data)}")
        print(f"pkl 数据长度: {len(data) if hasattr(data, '__len__') else 'N/A'}")

# 检查图片目录
img_prefix = cfg.data.train.img_prefix
print(f"\n图片目录是否存在: {os.path.exists(img_prefix)}")
if os.path.exists(img_prefix):
    files = os.listdir(img_prefix)
    print(f"图片目录下文件总数: {len(files)}")
    if len(files) > 0:
        print(f"前5个文件: {files[:5]}")

# 尝试构建训练集
print("\n" + "=" * 50)
print("正在尝试加载训练集...")
try:
    train_dataset = build_dataset(cfg.data.train)
    print(f"成功！训练集共有 {len(train_dataset)} 张图片")
except Exception as e:
    print(f"构建数据集时报错: {e}")
