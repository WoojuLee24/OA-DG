_base_ = [
    '/ws/external/configs/OA-DG/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes.py',
]

num_views = 2

""" OA-Loss configuration """
lw_jsd_rpn = 0.1
lw_jsd_roi = 10
lw_cont = 0.01
temperature = 0.06
random_proposal_cfg = dict(bbox_from='oagrb', num_bboxes=10, scales=(0.01, 0.3), ratios=(0.3, 1 / 0.3), iou_max=0.7, iou_min=0.0)

#############
### MODEL ###
#############
model = dict(
    rpn_head=dict(
        loss_cls=dict(
            type='CrossEntropyLossPlus', use_sigmoid=True, loss_weight=1.0, num_views=num_views,
            additional_loss='jsdv1_3_2aug', lambda_weight=lw_jsd_rpn, wandb_name='rpn_cls'),
        loss_bbox=dict(
            type='L1LossPlus', loss_weight=1.0, num_views=num_views,
            additional_loss="None", lambda_weight=0.0, wandb_name='rpn_bbox')),
    roi_head=dict(
        type='ContrastiveRoIHead',
        bbox_head=dict(
            type='Shared2FCContrastiveHead',
            with_cont=True,
            cont_predictor_cfg=dict(num_linear=2, feat_channels=256, return_relu=True),
            out_dim_cont=256,
            loss_cls=dict(
                type='CrossEntropyLossPlus', use_sigmoid=False, loss_weight=1.0, num_views=num_views,
                additional_loss='jsdv1_3_2aug', lambda_weight=lw_jsd_roi, wandb_name='roi_cls', log_pos_ratio=True),
            loss_bbox=dict(
                type='SmoothL1LossPlus', beta=1.0, loss_weight=1.0, num_views=num_views,
                additional_loss="None", lambda_weight=0.0, wandb_name='roi_bbox'),
            loss_cont=dict(
                type='ContrastiveLossPlus',
                loss_weight=lw_cont, num_views=num_views, temperature=temperature)
        ),
    ),
    train_cfg=dict(random_proposal_cfg=random_proposal_cfg)
)


""" OA-Mix configuration """
oamix_config=dict(
    type='OAMix', version='augmix',
    num_views=num_views, keep_orig=True, severity=10,
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
    dict(type='Collect', keys=['img', 'img2', 'gt_bboxes', 'gt_bboxes2', 'gt_labels',
                               'multilevel_boxes', 'oamix_boxes']),
]
data = dict(train=dict(dataset=dict(pipeline=train_pipeline)))