_base_ = [
    '/ws/external/configs/_base_/models/faster_rcnn_r50_fpn.py',
    '/ws/external/configs/_base_/datasets/cityscapes_detection.py',
    '/ws/external/configs/_base_/default_runtime.py'
]


#############
### MODEL ###
#############
model = dict(
    backbone=dict(init_cfg=None),
    roi_head=dict(
        bbox_head=dict(
            num_classes=8,
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rcnn=dict(dropout=False),
        wandb=dict(log=dict(features_list=[], vars=['log_vars'])),
    )
)


###############
### DATASET ###
###############
data = dict(samples_per_gpu=2, workers_per_gpu=4)

################
### RUN TIME ###
################
# optimizer
# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # [1] yields higher performance than [0]
    step=[1])
runner = dict(
    type='EpochBasedRunner', max_epochs=2) # # actual epoch = 2 * 8 = 16


###############
### runtime ###
###############
# yapf:disable
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
# yapf:enable
custom_hooks = []

# For better, more stable performance initialize from COCO
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa
