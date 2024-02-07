# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import os.path

import mmcv
from mmcv.image import tensor2imgs
import torch
from mmcv import DictAction
from mmcv.runner import (get_dist_info, init_dist)

from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataloader, build_dataset
import numpy as np
import PIL


''''
How 2 run?
    `python3 get_corrupted_dataset.py $CONFIG_FILE --show-dir $SAVE_DIR`
    E.g., 
        ```
        python3 get_corrupted_dataset.py \
        /ws/external/configs/generate_dataset/pascal_voc-c.py \
        --show-dir /ws/data/VOCdevkit-C/VOC2007 \
        --corruptions benchmark --seed 0
        ```
'''

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--corruptions',
        type=str,
        nargs='+',
        default='benchmark',
        choices=[
            'all', 'benchmark', 'noise', 'blur', 'weather', 'digital',
            'holdout', 'None', 'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow',
            'frost', 'fog', 'brightness', 'contrast', 'elastic_transform',
            'pixelate', 'jpeg_compression', 'speckle_noise', 'gaussian_blur',
            'spatter', 'saturate'
        ],
        help='corruptions')
    parser.add_argument(
        '--severities',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 4, 5],
        help='corruption severity levels')
    parser.add_argument(
        '--workers', type=int, default=32, help='workers per gpu')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def save_data(data_loader, out_dir=None):
    if 'cityscapes' in data_loader.dataset.img_prefix:
        dataset_type = 'cityscapes'
    elif 'coco' in data_loader.dataset.img_prefix:
        dataset_type = 'coco'
    elif 'VOCdevkit' in data_loader.dataset.img_prefix:
        dataset_type = 'pascal_voc'
    else:
        raise TypeError

    for i, data in enumerate(data_loader):
        img_metas = data['img_metas'].data[0]
        ori_filename = img_metas[0]['ori_filename']
        fn_ = ori_filename.split("/")
        if len(fn_) > 1:
            directory_name = '/'.join(fn_[:-1])
            if not os.path.exists(f"{out_dir}/{directory_name}"):
                os.makedirs(f"{out_dir}/{directory_name}")
        save_path = f'{out_dir}/{ori_filename}'

        if '.jpg' in save_path:
            save_path = save_path.replace('.jpg', '.png')

        img_tensor = data['img']
        img_vis = np.asarray(img_tensor[0], dtype=np.uint8)
        img = PIL.Image.fromarray(img_vis)
        img.save(save_path, quality='keep')

        print(save_path)

    return 0


def main():
    args = parse_args()

    assert args.show_dir, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "show-dir"')

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.workers == 0:
        args.workers = cfg.data.workers_per_gpu
    if args.debug:
        args.workers = 0
        args.launcher = 'none'

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed)

    if 'all' in args.corruptions:
        corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate',
            'jpeg_compression', 'speckle_noise', 'gaussian_blur', 'spatter',
            'saturate'
        ]
    elif 'benchmark' in args.corruptions:
        corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
            'brightness', 'contrast', 'elastic_transform', 'pixelate',
            'jpeg_compression'
        ]
    elif 'noise' in args.corruptions:
        corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise']
    elif 'blur' in args.corruptions:
        corruptions = [
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'
        ]
    elif 'weather' in args.corruptions:
        corruptions = ['snow', 'frost', 'fog', 'brightness']
    elif 'digital' in args.corruptions:
        corruptions = [
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]
    elif 'holdout' in args.corruptions:
        corruptions = ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
    elif 'None' in args.corruptions:
        corruptions = ['None']
        args.severities = [0]
    else:
        corruptions = args.corruptions

    rank, _ = get_dist_info()
    aggregated_results = {}
    for corr_i, corruption in enumerate(corruptions):
        aggregated_results[corruption] = {}
        for sev_i, corruption_severity in enumerate(args.severities):
            # evaluate severity 0 (= no corruption) only once
            if corr_i > 0 and corruption_severity == 0:
                continue
            test_data_cfg = copy.deepcopy(cfg.data.test)
            # assign corruption and severity

            if corruption_severity > 0:
                corruption_trans = dict(
                    type='Corrupt',
                    corruption=corruption,
                    severity=corruption_severity)
                # TODO: hard coded "1", we assume that the first step is
                # loading images, which needs to be fixed in the future
                test_data_cfg['pipeline'].insert(1, corruption_trans)

            # print info
            print(f'\nTesting {corruption} at severity {corruption_severity}')

            # build the dataloader
            # TODO: support multiple images per gpu
            #       (only minor changes are needed)
            dataset = build_dataset(test_data_cfg)
            data_loader = build_dataloader(
                dataset,
                samples_per_gpu=1,
                workers_per_gpu=args.workers,
                dist=distributed,
                shuffle=False)

            if not distributed:
                show_dir = args.show_dir

                show_dir = osp.join(show_dir, corruption)
                show_dir = osp.join(show_dir, str(corruption_severity))
                if not osp.exists(show_dir):
                    os.makedirs(show_dir)

                save_data(data_loader, out_dir=show_dir)
            else:
                raise TypeError("It does not support distribution mode")


if __name__ == '__main__':
    main()
