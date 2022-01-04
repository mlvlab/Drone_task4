_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', # ver1~ver7 : finetuning, version_1~
    # '../_base_/models/faster_rcnn_r50_fpn_person.py',
    '../_base_/datasets/triage_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'

# model = dict(
#     pretrained=None,
#     # triage_rpn_head=dict(
#     #     type='Triage_RPNHead',
#     #     in_channels=256,
#     #     feat_channels=256,
#     #     anchor_generator=dict(
#     #         type='AnchorGenerator',
#     #         scales=[8],
#     #         ratios=[0.5, 1.0, 2.0],
#     #         strides=[4, 8, 16, 32, 64]),
#     #     bbox_coder=dict(
#     #         type='DeltaXYWHBBoxCoder',
#     #         target_means=[.0, .0, .0, .0],
#     #         target_stds=[1.0, 1.0, 1.0, 1.0]),
#     #     loss_cls=dict(
#     #         type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
#     #     loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
#     triage_roi_head=dict(
#         type='Triage_StandardRoIHead',
#         bbox_roi_extractor=dict(
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
#             out_channels=256,
#             featmap_strides=[4, 8, 16, 32]),
#         bbox_head=dict(
#             type='Shared2FCBBoxHead',
#             in_channels=256,
#             fc_out_channels=1024,
#             roi_feat_size=7,
#             num_classes=4,
#             bbox_coder=dict(
#                 type='DeltaXYWHBBoxCoder',
#                 target_means=[0., 0., 0., 0.],
#                 target_stds=[0.1, 0.1, 0.2, 0.2]),
#             reg_class_agnostic=False,
#             loss_cls=dict(
#                 type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#             loss_bbox=dict(type='L1Loss', loss_weight=1.0)))
# )


# ver1~ver7 : finetuning
model = dict(
    pretrained=None,
    roi_head=dict(
        # type='StandardRoIHead',
        # bbox_roi_extractor=dict(
        #     type='SingleRoIExtractor',
        #     roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
        #     out_channels=256,
        #     featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=4, # class 0 to 3
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                # type='GIoULoss', loss_weight=1.0),
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)))
)






# GIoULoss
# config_file = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'


# optimizer
# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001) #: version_1 work_dirs/triage_examine1600_version_1
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005) #: version_2 work_dirs/triage_examine1600_version_2
#optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001) #: version_3 work_dirs/triage_examine1600_version_3
#optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005) #: version_4 work_dirs/triage_examine1600_version_4
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7])
# the max_epochs and step in lr_config need specifically tuned for the customized dataset
runner = dict(max_epochs=30)
log_config = dict(interval=100)
