_base_ = [
    '../_base_/models/vfnet.py',
    '../_base_/datasets/triage_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

checkpoint_file = 'checkpoints/vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth'
# load_from = 'checkpoints/vfnet_r50_fpn_1x_coco_20201027-38db6f58.pth'

model = dict(
    pretrained=None,
    bbox_head=dict(
        type='VFNetHead',
        num_classes=4,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=False,
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)),
)



# optimizer
optimizer = dict(
    lr=0.01, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=30)