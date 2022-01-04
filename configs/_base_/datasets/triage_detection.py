# dataset settings
dataset_type = 'MyTriageDataset'
# data_root = 'data/our_dataset/triage/onlyhandmade/1023/'
data_root = 'our_dataset/final_triage/' # version_1~4
# data_root = 'our_dataset/examine_1600_triage/'
# data_root = 'our_dataset/validset/'

# data_root = 'our_dataset/personTriage/'

# data_root = 'data/our_dataset/triage/1102_-1to3_500/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    # dict(type='Resize', img_scale=(, 416), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomShift', shift_ratio=0.2),
    dict(type='Corrupt', corruption='gaussian_noise', severity=2),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32), # padding 
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    # dict(type='Shearing')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        # img_scale=(416, 416),
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
        ann_file=data_root + 'train/_annotations.coco.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='our_dataset/validset/valid/_annotations.coco.json',
        img_prefix='our_dataset/validset/valid/',
        # ann_file='our_dataset/valid_5class/valid/_annotations.coco.json',
        # img_prefix='our_dataset/valid_5class/valid/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='our_dataset/validset/valid/_annotations.coco.json',
        img_prefix='our_dataset/validset/valid/',
        # ann_file='our_dataset/valid_5class/valid/_annotations.coco.json',
        # img_prefix='our_dataset/valid_5class/valid/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
