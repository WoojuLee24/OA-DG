import copy

import numpy as np
# from numpy import random
import random

from ..builder import PIPELINES

import cv2

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

from PIL import Image, ImageDraw, ImageFilter
from mmdet.datasets.pipelines.augmix import (autocontrast, equalize, posterize, solarize, color,
                                             contrast, brightness, sharpness, invert,
                                             rotate, shear_x, shear_y, translate_x, translate_y,)


# BBoxOnlyAugmentation
# REF: https://github.com/poodarchu/learn_aug_for_object_detection.numpy/
def _apply_bbox_only_augmentation(img, bbox_xy, aug_func, fillmode=None, fillcolor=None, return_bbox=False, radius=10,
                                  radius_ratio=None, margin=3, sigma_ratio=None, times=3, blur_bbox=None, **kwargs):
    '''
    Args:
        img     : (np.array) (img_width, img_height, channel)
        bbox_xy : (tensor) [x1, y1, x2, y2]
        aug_func: (func) can be contain 'level', 'img_size', etc.
    '''
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)

    # Get bbox_content from image
    img_height, img_width = img.shape[0], img.shape[1]
    x1, y1, x2, y2 = int(bbox_xy[0]), int(bbox_xy[1]), int(bbox_xy[2]), int(bbox_xy[3])
    if (x2-x1) < 1 or (y2-y1) < 1:
        return (np.asarray(img, dtype=np.uint8), bbox_xy) if return_bbox \
            else np.asarray(img, dtype=np.uint8)

    # bbox_content
    bbox_content = img
    kwargs['img_size'] = (img_width, img_height)

    center = ((x1 + x2) / 2., (y1 + y2) / 2.)
    kwargs['img_size_for_level'] = (x2-x1+1, y2-y1+1)

    # Augment
    outputs = aug_func(bbox_content, **kwargs, fillcolor=fillcolor, center=center,
                       bbox_xy=bbox_xy, return_bbox=return_bbox)

    augmented_bbox_content = np.asarray(outputs['img'])
    augmented_gt_bbox = outputs['gt_bbox'] if 'gt_bbox' in outputs else bbox_xy

    mask = 1.0 - blur_bbox

    # Overwrite augmented_bbox_content into img
    img = img * mask + augmented_bbox_content * (1.0 - mask)

    if return_bbox:
        return np.asarray(img, dtype=np.uint8), augmented_gt_bbox
    else:
        return np.asarray(img, dtype=np.uint8)


def _apply_bboxes_only_augmentation(img, bboxes_xy, aug_func, mask_bboxes=None, **kwargs):
    '''
    Args:
        img         : (np.array) (img_width, img_height, channel)
        bboxes_xy   : (tensor) has shape of (num_bboxes, 4) with [x1, y1, x2, y2]
        aug_func    : (func) The argument is bbox_content # TODO: severity?
    '''
    assert len(bboxes_xy) == len(mask_bboxes)
    for i in range(len(bboxes_xy)):
        blur_bbox = None if mask_bboxes is None else mask_bboxes[i] # PIL.Image.fromarray(np.asarray(mask_bboxes[i] * 255, dtype=np.uint8))
        img = _apply_bbox_only_augmentation(img, bboxes_xy[i], aug_func, blur_bbox=blur_bbox, **kwargs)
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    else:
        return img


def bboxes_only_rotate(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bboxes_only_augmentation(pil_img, bboxes_xy, rotate, level=level, img_size=img_size, **kwargs)


def bboxes_only_shear_x(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bboxes_only_augmentation(pil_img, bboxes_xy, shear_x, level=level, img_size=img_size, **kwargs)


def bboxes_only_shear_y(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bboxes_only_augmentation(pil_img, bboxes_xy, shear_y, level=level, img_size=img_size, **kwargs)


def bboxes_only_shear_xy(pil_img, bboxes_xy, level, img_size, **kwargs):
    func = bboxes_only_shear_x if np.random.rand() < 0.5 else bboxes_only_shear_y
    return func(pil_img, bboxes_xy, level, img_size, **kwargs)


def bboxes_only_translate_x(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bboxes_only_augmentation(pil_img, bboxes_xy, translate_x, level=level, img_size=img_size, **kwargs)


def bboxes_only_translate_y(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bboxes_only_augmentation(pil_img, bboxes_xy, translate_y, level=level, img_size=img_size, **kwargs)


def bboxes_only_translate_xy(pil_img, bboxes_xy, level, img_size, **kwargs):
    func = bboxes_only_translate_x if np.random.rand() < 0.5 else bboxes_only_translate_y
    return func(pil_img, bboxes_xy, level, img_size, **kwargs)


# Random bboxes only augmentation
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
def generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy=None,
                              scales=(0.01, 0.2), ratios=(0.3, 1/0.3),
                              max_iters=100, eps=1e-6, allow_fg=False, **kwargs):
    # REF: mmdetection/mmdet/datasets/pipelines/transforms.py Cutout
    if isinstance(num_bboxes, tuple) or isinstance(num_bboxes, list):
        num_bboxes = np.random.randint(num_bboxes[0], num_bboxes[1] + 1)
    (img_width, img_height) = img_size

    random_bboxes_xy = np.zeros((num_bboxes, 4))
    total_bboxes = 0
    for i in range(max_iters):
        if total_bboxes >= num_bboxes:
            break

        # Generate a random bbox.
        x1, y1 = np.random.randint(0, img_width), np.random.randint(0, img_height)
        scale = np.random.uniform(*scales) * img_height * img_width
        ratio = np.random.uniform(*ratios)
        bbox_w, bbox_h = int(np.sqrt(scale / ratio)), int(np.sqrt(scale * ratio))
        random_bbox = np.array([[x1, y1, min(x1 + bbox_w, img_width), min(y1 + bbox_h, img_height)]])
        if bboxes_xy is not None:
            ious = bbox_overlaps(random_bbox, bboxes_xy)
            if np.sum(ious) > eps:
                if allow_fg:
                    diff_bboxes = random_bbox - bboxes_xy
                    diff_bboxes[:, :2] = (diff_bboxes[:, :2] > 0)
                    diff_bboxes[:, 2:] = (diff_bboxes[:, 2:] < 0)
                    diff_mask = diff_bboxes.sum(axis=1) < 4
                    if diff_mask.all():
                        continue
        random_bboxes_xy[total_bboxes, :] = random_bbox[0]
        total_bboxes += 1
    if total_bboxes != num_bboxes:
        random_bboxes_xy = random_bboxes_xy[:total_bboxes, :]

    return random_bboxes_xy


def random_bboxes_only_rotate(pil_img, bboxes_xy, level, img_size, num_bboxes, return_bbox=False, **kwargs):
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, **kwargs)
    if 'blur_bboxes' in kwargs:
        kwargs['blur_bboxes'] = None
    return _apply_bboxes_only_augmentation(pil_img, random_bboxes_xy, rotate, level=level, img_size=img_size, return_bbox=False, **kwargs)


def random_bboxes_only_shear_x(pil_img, bboxes_xy, level, img_size, num_bboxes, return_bbox=False, **kwargs):
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, return_bbox=False, **kwargs)
    if 'blur_bboxes' in kwargs:
        kwargs['blur_bboxes'] = None
    return _apply_bboxes_only_augmentation(pil_img, random_bboxes_xy, shear_x, level=level, img_size=img_size, return_bbox=False, **kwargs)


def random_bboxes_only_shear_y(pil_img, bboxes_xy, level, img_size, num_bboxes, return_bbox=False, **kwargs):
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, return_bbox=False, **kwargs)
    if 'blur_bboxes' in kwargs:
        kwargs['blur_bboxes'] = None
    return _apply_bboxes_only_augmentation(pil_img, random_bboxes_xy, shear_y, level=level, img_size=img_size, return_bbox=False, **kwargs)


def random_bboxes_only_shear_xy(pil_img, bboxes_xy, level, img_size, num_bboxes, return_bbox=False, **kwargs):
    func = bboxes_only_shear_x if np.random.rand() < 0.5 else bboxes_only_shear_y
    if 'blur_bboxes' in kwargs:
        kwargs['blur_bboxes'] = None
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, return_bbox=False, **kwargs)
    return func(pil_img, random_bboxes_xy, level, img_size, **kwargs)


def random_bboxes_only_translate_x(pil_img, bboxes_xy, level, img_size, num_bboxes, return_bbox=False, **kwargs):
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, return_bbox=False, **kwargs)
    if 'blur_bboxes' in kwargs:
        kwargs['blur_bboxes'] = None
    return _apply_bboxes_only_augmentation(pil_img, random_bboxes_xy, translate_x, level=level, img_size=img_size, return_bbox=False, **kwargs)


def random_bboxes_only_translate_y(pil_img, bboxes_xy, level, img_size, num_bboxes, return_bbox=False, **kwargs):
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, return_bbox=False, **kwargs)
    if 'blur_bboxes' in kwargs:
        kwargs['blur_bboxes'] = None
    return _apply_bboxes_only_augmentation(pil_img, random_bboxes_xy, translate_y, level=level, img_size=img_size, return_bbox=False, **kwargs)


def random_bboxes_only_translate_xy(pil_img, bboxes_xy, level, img_size, num_bboxes, return_bbox=False, **kwargs):
    func = bboxes_only_translate_x if np.random.rand() < 0.5 else bboxes_only_translate_y
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, return_bbox=False, **kwargs)
    if 'blur_bboxes' in kwargs:
        kwargs['blur_bboxes'] = None
    return func(pil_img, random_bboxes_xy, level, img_size, **kwargs)


# Random bboxes + ground-truth bboxes only augmentation
def random_gt_only_rotate(pil_img, bboxes_xy, level, img_size, num_bboxes, sample_gt_ratio=1, **kwargs):
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, **kwargs)
    num_gt_samples = int(len(bboxes_xy)*sample_gt_ratio) if len(bboxes_xy) > 1 else len(bboxes_xy)
    gt_bboxes_xy = np.stack(random.sample(list(bboxes_xy), num_gt_samples))
    random_gt_bboxes_xy = np.concatenate([random_bboxes_xy, gt_bboxes_xy], axis=0)
    return _apply_bboxes_only_augmentation(pil_img, random_gt_bboxes_xy, rotate, level=level, img_size=img_size, **kwargs)


def random_gt_only_shear_xy(pil_img, bboxes_xy, level, img_size, num_bboxes, sample_gt_ratio=1, **kwargs):
    func = bboxes_only_shear_x if np.random.rand() < 0.5 else bboxes_only_shear_y
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, **kwargs)
    num_gt_samples = int(len(bboxes_xy) * sample_gt_ratio) if len(bboxes_xy) > 1 else len(bboxes_xy)
    gt_bboxes_xy = np.stack(random.sample(list(bboxes_xy), num_gt_samples))
    random_gt_bboxes_xy = np.concatenate([random_bboxes_xy, gt_bboxes_xy], axis=0)
    return func(pil_img, random_gt_bboxes_xy, level, img_size, **kwargs)


def random_gt_only_translate_xy(pil_img, bboxes_xy, level, img_size, num_bboxes, sample_gt_ratio=1, **kwargs):
    func = bboxes_only_translate_x if np.random.rand() < 0.5 else bboxes_only_translate_y
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, **kwargs)
    num_gt_samples = int(len(bboxes_xy) * sample_gt_ratio) if len(bboxes_xy) > 1 else len(bboxes_xy)
    gt_bboxes_xy = np.stack(random.sample(list(bboxes_xy), num_gt_samples))
    random_gt_bboxes_xy = np.concatenate([random_bboxes_xy, gt_bboxes_xy], axis=0)
    return func(pil_img, random_gt_bboxes_xy, level, img_size, **kwargs)


# Background only augmentation
def _apply_bg_only_augmentation(img, bboxes_xy, aug_func, mask_bboxes=None, fillmode=None,
                                fillcolor=0, return_bbox=False, radius=10,
                                radius_ratio=None, bg_margin=3, times=3, margin_bg=False, sigma_ratio=None, blur_bboxes=None, **kwargs):
    '''
    Args:
        img         : (np.array) (img_width, img_height, channel)
        bboxes_xy   : (tensor) has shape of (num_bboxes, 4) with [x1, y1, x2, y2]
        aug_func    : (func) The argument is bbox_content # TODO: severity?
    '''
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)

    # Augment
    bbox_content = img.copy()
    img_shape = img.copy().shape
    kwargs['img_size'] = (img_shape[1], img_shape[0])

    if mask_bboxes is None or len(mask_bboxes) == 0:
        mask = np.zeros_like(img)
    else:
        mask = np.max(mask_bboxes, axis=0)

    # Overwrite augmented_bbox_content into img
    outputs = aug_func(bbox_content, return_bbox=False, **kwargs, fillcolor=fillcolor,
                       mask=np.asarray(mask * 255, dtype=np.uint8))

    augmented_bbox_content = outputs['img']
    augmented_mask = np.asarray(outputs['mask']) / 255

    maintained_mask = np.maximum(mask, augmented_mask)
    img = maintained_mask * img + (1.0 - maintained_mask) * augmented_bbox_content

    return np.asarray(img, dtype=np.uint8)


def bg_only_rotate(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bg_only_augmentation(pil_img, bboxes_xy, rotate, level=level, img_size=img_size, **kwargs)


def bg_only_shear_x(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bg_only_augmentation(pil_img, bboxes_xy, shear_x, level=level, img_size=img_size, **kwargs)


def bg_only_shear_y(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bg_only_augmentation(pil_img, bboxes_xy, shear_y, level=level, img_size=img_size, **kwargs)


def bg_only_shear_xy(pil_img, bboxes_xy, level, img_size, **kwargs):
    func = bg_only_shear_x if np.random.rand() < 0.5 else bg_only_shear_y
    return func(pil_img, bboxes_xy, level, img_size, **kwargs)


def bg_only_translate_x(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bg_only_augmentation(pil_img, bboxes_xy, translate_x, level=level, img_size=img_size, **kwargs)


def bg_only_translate_y(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bg_only_augmentation(pil_img, bboxes_xy, translate_y, level=level, img_size=img_size, **kwargs)


def bg_only_translate_xy(pil_img, bboxes_xy, level, img_size, **kwargs):
    func = bg_only_translate_x if np.random.rand() < 0.5 else bg_only_translate_y
    return func(pil_img, bboxes_xy, level, img_size, **kwargs)

