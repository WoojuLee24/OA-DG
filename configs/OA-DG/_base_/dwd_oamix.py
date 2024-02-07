_base_ = [
    '/ws/external/configs/_base_/datasets/s-dgod.py',
]

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
    dict(type='Collect', keys=['img', 'img2', 'gt_bboxes', 'gt_bboxes2', 'gt_labels',
                               'multilevel_boxes', 'oamix_boxes']),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        dataset=dict(
            pipeline=train_pipeline)),)
