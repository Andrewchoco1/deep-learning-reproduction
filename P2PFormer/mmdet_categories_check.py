from mmdet.datasets import build_dataset
from mmcv import Config

cfg = Config.fromfile('你的配置文件.py')

import json
with open(cfg.data.train.ann_file, 'r') as f:
    coco_data = json.load(f)

print(f"标注中的类别: {coco_data['categories']}")
print(f"WhuDataset 定义的类别: {'building'}")  # 这是问题所在

