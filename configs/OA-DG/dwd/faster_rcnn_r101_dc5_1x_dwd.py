_base_ = [
    '/ws/external/configs/_base_/models/faster_rcnn_r50_caffe_dc5.py',
    '/ws/external/configs/_base_/datasets/s-dgod.py',
    '/ws/external/configs/_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet101_caffe')),
    roi_head=dict(bbox_head=dict(num_classes=7)))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)    # original: 0.01
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = * 2 => 8 , 16
lr_config = dict(policy='step', step=[4, 8])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=10)  # actual epoch = 10 * 2 = 20
