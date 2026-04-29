# Deep-Learning-Reproduction

## Overview
This repository contains personal re-implementation and reproduction of classic deep learning and computer vision models.

All codes are developed with PyTorch, focusing on understanding network principles and practicing model construction from scratch.

## Purpose
- Build neural networks manually to consolidate basic coding skills
- Get familiar with the complete training, validation and inference pipeline
- Learn classic CNN, attention and Transformer-based vision models
- Lay a solid foundation for future large model research and engineering projects

## Debugging Notes
[P2PFormer](https://github.com/zhang-tao-whu/P2PFormer/)
<details>
<summary>🔧 : "模型不训练，直接进入验证集" 问题排查</summary>

### 现象
更换数据集路径后，模型跳过训练直接进入验证阶段，但两个数据集格式完全一致。

### 排查过程

| 步骤 | 假设 | 验证方法 | 结论 |
|------|------|----------|------|
| 1 | 路径拼接错误 | 打印配置中的 `ann_file`、`img_prefix`，检查文件存在性 | ❌ 路径正确，所有文件可访问 |
| 2 | 环境/模块未注册 | `import p2pformer` | ❌ 环境正常 |
| 3 | 数据集加载异常 | 单独执行 `build_dataset`，打印训练集长度 | ✅ 训练集加载了 **0 张图片** |
| 4 | 文件名不匹配 | `os.path.exists` 逐个验证标注中的文件名 | ❌ 文件名全部存在 |
| 5 | 数据集类内部过滤 | 查看 `WhuDataset` 源码 | ✅ **根因定位** |

### 根本原因

`WhuDataset` 继承自 mmdet 的 `CocoDataset`，该类在初始化时会进行**类别名匹配**：将标注文件 `categories` 中的类别名与数据集类 `CLASSES` 元组中的类别名逐一比对。不匹配的类别及其所有标注框会被静默丢弃，导致训练集图片被全部过滤。

```python
@DATASETS.register_module()
class WhuDataset(CocoDataset):
    CLASSES = ('building', )  # 旧数据集
    # 新数据集标注中的类别是 'Farmland' → 全部标注被丢弃
