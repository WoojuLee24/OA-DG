_base_ = [
    '/ws/external/configs/_base_/models/faster_rcnn_r50_caffe_dc5.py',
    '/ws/external/configs/_base_/default_runtime.py',
    '/ws/external/configs/_base_/datasets/s-dgod.py',
]

name = f"r101_oamixall20.8"


#############
### MODEL ###
#############
num_views=1
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='open-mmlab://detectron2/resnet101_caffe')),
    rpn_head=dict(
            loss_cls=dict(
                type='CrossEntropyLossPlus', use_sigmoid=True, loss_weight=1.0, num_views=num_views,
                additional_loss='None', lambda_weight=0.0, wandb_name='rpn_cls'),
            loss_bbox=dict(type='L1LossPlus', loss_weight=1.0, num_views=num_views,
                           additional_loss="None", lambda_weight=0.0001, wandb_name='rpn_bbox')),
    roi_head=dict(
        bbox_head=dict(
            num_classes=7,
            loss_cls=dict(
                type='CrossEntropyLossPlus', use_sigmoid=False, loss_weight=1.0, num_views=num_views,
                additional_loss='None', lambda_weight=0.0, wandb_name='roi_cls', log_pos_ratio=True),
            loss_bbox=dict(type='SmoothL1LossPlus', beta=1.0, loss_weight=1.0, num_views=num_views,
                           additional_loss="None", lambda_weight=0.0001, wandb_name='roi_bbox'),
        ),
    ),
    train_cfg=dict(
        wandb=dict(
            log=dict(
                features_list=[],
                vars=['log_vars']),
        ),
    ))

oamix_config=dict(
    type='OAMix',
    version='augmix.all',
    use_mix=True, mixture_width=1, mixture_depth=-1,
    use_oa=True, oa_version='saliency_sparse', use_mrange=False,
    use_multilevel=True,
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
    dict(
        type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    oamix_config,
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels',
                               ]),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        dataset=dict(
            pipeline=train_pipeline)),)



################
### RUN TIME ###
################
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)    # original: 0.01
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = * 2 => 8 , 16
lr_config = dict(policy='step', step=[4, 8])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=10)  # actual epoch = 10 * 2 = 20

###########
### LOG ###
###########
custom_hooks = [
    dict(type='FeatureHook',
         layer_list=model['train_cfg']['wandb']['log']['features_list']),
]

print('++++++++++++++++++++')
print(f"{name}")
print('++++++++++++++++++++')

log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])