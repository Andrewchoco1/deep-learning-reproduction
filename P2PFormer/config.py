#put your config
# dataset settings
dataset_type = 'WhuDataset'
data_root = '/mnt/vde/FW/dataset_all/scp/WHU'
classes = ('building')
img_norm_cfg = dict(
    mean=[128.406, 130.338, 121.896], std=[61.104, 57.831, 61.201], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(1344, 816), (1344, 1344)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='AlignSampleBoundary', point_nums=128, reset_bbox=True),
    dict(type='ContourDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels',
         'gt_masks', 'gt_polys', 'key_points_masks', 'key_points']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/train_patches_512.json',
        img_prefix=data_root + '/train/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/test_patches_512.json',
        img_prefix=data_root + '/test/images/',
        pipeline=test_pipeline),
    #test=dict(
    #    type=dataset_type,
    #    ann_file=data_root + 'annotations/train_val.json',
    #    img_prefix=data_root + 'train_val/',
    #    pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/test_patches_512.json',
        img_prefix=data_root + '/test/images/',
        pipeline=test_pipeline),
    )
evaluation = dict(metric=['bbox', 'segm'], interval=20)
