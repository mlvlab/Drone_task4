#_base_ = [
#    '../_base_/models/faster_rcnn_r50_fpn.py',
#    '../_base_/datasets/person_detection.py',
#    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
#]

_base_ = './faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py'
model = dict(roi_head=dict(bbox_head=dict(num_classes=1)))
#classes = ('person', )
#data = dict(
#    train=dict(classes=classes),
#    val=dict(classes=classes),
#    test=dict(classes=classes))


_base_ = [
    '../faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py',
    '../_base_/datasets/person_detection.py'
]

#checkpoint_file = 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_bbox_mAP-0.398_20200504_163323-30042637.pth'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'
model = dict(roi_head=dict(bbox_head=dict(num_classes=1)))

# optimizer
# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
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
