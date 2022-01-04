_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', 
   # '../_base_/datasets/triage_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
#dataset_type = 'CocoDataset'
dataset_type = 'MyPersonDataset'
classes = ('person',)
data_root = 'our_dataset/person_old/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Corrupt', corruption='gaussian_noise', severity=2),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
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
        classes = classes,
        ann_file=data_root + 'train/_annotations.coco.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/set01_test/test/_annotations.coco.json',
        img_prefix='data/set01_test/test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='data/set01_test/test/_annotations.coco.json',
        img_prefix='data/set01_test/test/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')



checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'

model = dict(
#    pretrained=None,
    roi_head=dict(
        # type='StandardRoIHead',
        # bbox_roi_extractor=dict(
        #     type='SingleRoIExtractor',
        #     roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
        #     out_channels=256,
        #     featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            #type='Shared2FCBBoxHead',
            #in_channels=256,
            #fc_out_channels=1024,
            #roi_feat_size=7,
            num_classes=1, # class 0 to 3
            #bbox_coder=dict(
            #    type='DeltaXYWHBBoxCoder',
            #   target_means=[0., 0., 0., 0.],
            #    target_stds=[0.1, 0.1, 0.2, 0.2]),
            #reg_class_agnostic=False,
            #loss_cls=dict(
                # type='GIoULoss', loss_weight=1.0),
            #    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            #loss_bbox=dict(type='L1Loss', loss_weight=1.0)
            )
        )
)






# GIoULoss
# config_file = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'


# optimizer
# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001) #: version_1 w
#optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005) #: version_2 work_dirs/triage1600_version2
#optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001) #: version_3 work_dirs/triage1600_version3
#optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005) #: version_4 work_dirs/triage1600_version4
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7])
# the max_epochs and step in lr_config need specifically tuned for the customized dataset
runner = dict(max_epochs=10)
log_config = dict(interval=100)
