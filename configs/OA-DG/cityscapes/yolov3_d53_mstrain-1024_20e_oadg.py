_base_ = [
    '/ws/external/configs/OA-DG/cityscapes/yolov3_d53_mstrain-1024_20e.py'
]

num_views = 2

""" OA-Loss configuration """
jsd_conf_weight = 20.0
jsd_cls_weight = 5.0
cont_cfg = dict(loss_weight=1.0, dim=256, temperature=0.06)

#############
### MODEL ###
#############
model = dict(
    bbox_head=dict(
        type='YOLOV3HeadCont',
        num_classes=8,
        jsd_conf_weight=jsd_conf_weight,
        jsd_cls_weight=jsd_cls_weight,
        cont_cfg=cont_cfg)
)


""" OA-Mix configuration """
oamix_config = dict(
    type='OAMix', version='augmix',
    num_views=num_views, keep_orig=False, severity=10,
    # multi-level transformation
    random_box_ratio=(3, 1/3), random_box_scale=(0.01, 0.1),
    # object-aware mixing # scales, ratios
    oa_random_box_scale=(0.005, 0.1), oa_random_box_ratio=(3, 1 / 3),
    spatial_ratio=4, sigma_ratio=0.3, # smoothing strategy to improve speed
)

#################
#### DATASET #####
#################
custom_imports = dict(imports=['mmdet.datasets.pipelines.oa_mix'], allow_failed_imports=False)
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=[(800, 800), (1024, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    oamix_config,
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img2', 'gt_bboxes', 'gt_bboxes2', 'gt_labels',
                               'multilevel_boxes', 'oamix_boxes']),
]

data = dict(train=dict(pipeline=train_pipeline))
