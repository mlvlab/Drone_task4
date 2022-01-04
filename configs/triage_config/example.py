_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# import pdb; pdb.set_trace()
# dataset_type = 'CocoDataset'
dataset_type = 'MyPersonDataset'
classes = ('person')


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='dataset_type',
        classes=classes,
        # ann_file= '../../data/our_dataset/annotation/official_crop_coco/train/_annotations.coco.json',
        ann_file= 'data/our_dataset/annotation/person_dataset/train/_annotations.coco.json',
        img_prefix= 'data/our_dataset/annotation/person_dataset/train/',
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file= 'data/our_dataset/annotation/person_dataset/valid/_annotations.coco.json',
        img_prefix= 'data/our_dataset/annotation/person_dataset/valid/',
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file= 'data/our_dataset/annotation/person_dataset/test/_annotations.coco.json',
        img_prefix= 'data/our_dataset/annotation/person_dataset/test/',
    ))
# evaluation = dict(interval=1, metric='bbox')

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
    )
)

# config_file = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
