# Copyright (c) OpenMMLab. All rights reserved.
import os.path
import numpy as np
from PIL import Image

import mmcv
from mmcv.image import tensor2imgs


def save_data(data_loader, win_name='', out_dir=None, show_score_thr=0.3):
    results = []
    for i, data in enumerate(data_loader):
        img_tensor = data['img'][0] # .data[0]
        img_metas = data['img_metas'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])

        ori_filename = img_metas[0]['ori_filename']
        fn_ = ori_filename.split("/")
        directory_name, filename = fn_[0], fn_[1]
        if not os.path.exists(f"{out_dir}/{directory_name}"):
            os.makedirs(f"{out_dir}/{directory_name}")
        save_path = f'{out_dir}/{directory_name}/{filename}'

        img = imgs[0]
        img = mmcv.bgr2rgb(img)
        img = np.ascontiguousarray(img)
        img_save = Image.fromarray(img)
        img_save.save(save_path)

    return results

