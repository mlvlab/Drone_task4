_base_ = [
    # '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/triage_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]


model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet101_caffe')),
    roi_head=dict(bbox_head=dict(num_classes=4))
            )


checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'