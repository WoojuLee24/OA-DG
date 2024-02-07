_base_ = [
    '/ws/external/configs/OA-DG/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes.py'
]

oamix_config=dict(
    type='OAMix', version='augmix',
    num_views=1, keep_orig=False, severity=10,
    # multi-level transformation
    random_box_ratio=(3, 1/3), random_box_scale=(0.01, 0.1),
    # object-aware mixing # scales, ratios
    oa_random_box_scale=(0.005, 0.1), oa_random_box_ratio=(3, 1 / 3),
    spatial_ratio=4, sigma_ratio=0.3, # smoothing strategy to improve speed
)

###############
### DATASET ###
###############
custom_imports = dict(imports=['mmdet.datasets.pipelines.oa_mix'], allow_failed_imports=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    oamix_config,
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(train=dict(dataset=dict(pipeline=train_pipeline)))
