# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import math
import warnings

import cv2
import mmcv
import numpy as np
from numpy import random
import torch

from mmdet.core import PolygonMasks, find_inside_bboxes
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..builder import PIPELINES
from .compose import Compose

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

from PIL import Image, ImageOps, ImageEnhance

def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, **kwargs):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, **kwargs):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level, **kwargs):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def invert(pil_img, **kwargs):
  return ImageOps.invert(pil_img)

def rgb_to_gray(pil_img, **kwargs):
  return pil_img.convert('L').convert('RGB')

def rotate(pil_img, level, img_size, fillcolor=None, center=None, mask=None, bbox_xy=None, return_bbox=False, **kwargs):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees

    # img = pil_img.rotate(degrees, resample=Image.BILINEAR, fillcolor=fillcolor, center=center)
    if center is None:
        center = (img_size[0] / 2, img_size[1] / 2)
    M = cv2.getRotationMatrix2D(center, degrees, 1.0)
    outputs = dict(img=cv2.warpAffine(pil_img, M, img_size))

    if mask is not None:
        outputs['mask'] = cv2.warpAffine(mask, M, img_size) # mask.rotate(degrees, resample=Image.BILINEAR, fillcolor=fillcolor, center=center)

    if return_bbox:
        outputs['gt_bbox'] = bbox_xy

    return outputs


def solarize(pil_img, level, **kwargs):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level, img_size, fillcolor=None, center=None, mask=None, bbox_xy=None, return_bbox=False, **kwargs):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    tx = 0 if center is None else -level * center[1]

    # img = pil_img.transform(img_size, Image.AFFINE, (1, level, tx, 0, 1, 0), resample=Image.BILINEAR, fillcolor=fillcolor)
    M = np.float32([[1, -level, -tx], [0, 1, 0]])
    outputs = dict(img=cv2.warpAffine(pil_img, M, (0, 0)))

    if mask is not None:
        # outputs['mask'] = mask.transform(img_size, Image.AFFINE, (1, level, tx, 0, 1, 0), resample=Image.BILINEAR, fillcolor=fillcolor)
        outputs['mask'] = cv2.warpAffine(mask, M, (0, 0))

    if return_bbox:
        outputs['gt_bbox'] = bbox_xy

    return outputs


def shear_y(pil_img, level, img_size, fillcolor=None, center=None, mask=None, bbox_xy=None, return_bbox=False, **kwargs):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    ty = 0 if center is None else -level * center[0]

    # img = pil_img.transform(img_size, Image.AFFINE, (1, 0, 0, level, 1, ty), resample=Image.BILINEAR, fillcolor=fillcolor)
    M = np.float32([[1, 0, 0], [-level, 1, -ty]])
    outputs = dict(img=cv2.warpAffine(pil_img, M, (0, 0)))

    if mask is not None:
        # outputs['mask'] = mask.transform(img_size, Image.AFFINE, (1, 0, 0, level, 1, ty), resample=Image.BILINEAR, fillcolor=fillcolor)
        outputs['mask'] = cv2.warpAffine(mask, M, (0, 0))

    if return_bbox:
        outputs['gt_bbox'] = bbox_xy

    return outputs


def translate_x(pil_img, level, img_size, fillcolor=None, img_size_for_level=None, mask=None, bbox_xy=None, return_bbox=False, **kwargs):
    maxval = img_size[0] if img_size_for_level is None else img_size_for_level[0]
    level = int_parameter(sample_level(level), maxval / 3)
    if np.random.random() > 0.5:
        level = -level

    # img = pil_img.transform(img_size, Image.AFFINE, (1, 0, level, 0, 1, 0), resample=Image.BILINEAR, fillcolor=fillcolor)
    M = np.float32([[1, 0, -level], [0, 1, 0]])
    outputs = dict(img=cv2.warpAffine(pil_img, M, (0, 0)))

    if mask is not None:
        # outputs['mask'] = mask.transform(img_size, Image.AFFINE, (1, 0, level, 0, 1, 0), resample=Image.BILINEAR, fillcolor=fillcolor)
        outputs['mask'] = cv2.warpAffine(mask, M, (0, 0))
    if return_bbox:
        bbox_xy[0] = max(bbox_xy[0], bbox_xy[0] - level)
        bbox_xy[2] = min(bbox_xy[2], bbox_xy[2] - level)
        outputs['gt_bbox'] = bbox_xy

    return outputs


def translate_y(pil_img, level, img_size, fillcolor=None, img_size_for_level=None, mask=None, bbox_xy=None, return_bbox=False, **kwargs):
    maxval = img_size[1] if img_size_for_level is None else img_size_for_level[1]
    level = int_parameter(sample_level(level), maxval / 3)
    if np.random.random() > 0.5:
        level = -level

    # img = pil_img.transform(img_size, Image.AFFINE, (1, 0, 0, 0, 1, level), resample=Image.BILINEAR, fillcolor=fillcolor)
    M = np.float32([[1, 0, 0], [0, 1, -level]])
    outputs = dict(img=cv2.warpAffine(pil_img, M, (0, 0)))

    if mask is not None:
        # outputs['mask'] = mask.transform(img_size, Image.AFFINE, (1, 0, 0, 0, 1, level), resample=Image.BILINEAR, fillcolor=fillcolor)
        outputs['mask'] = cv2.warpAffine(mask, M, (0, 0))

    if return_bbox:
        bbox_xy[1] = max(bbox_xy[1], bbox_xy[1] - level)
        bbox_xy[3] = min(bbox_xy[3], bbox_xy[3] - level)
        outputs['gt_bbox'] = bbox_xy

    return outputs


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level, **kwargs):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level, **kwargs):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level, **kwargs):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level, **kwargs):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)




"""
Augmix with pillow
"""
@PIPELINES.register_module()
class AugMix:
    def __init__(self, mean, std, aug_list='augmentations', to_rgb=True, no_jsd=False, aug_severity=1,
                 num_views=3):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

        self.mixture_width = 3
        self.mixture_depth = -1

        self.aug_prob_coeff = 1.
        self.aug_severity = aug_severity

        self.no_jsd = no_jsd
        self.num_views = num_views

        augmentations = [
            autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
            translate_x, translate_y
        ]
        augmentations_all = [ # WARN: don't use it if dataset-c
            autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
            translate_x, translate_y, color, contrast, brightness, sharpness
        ]
        augmentations_without_obj_translation = [ # WARN: don't use it if dataset-c
            autocontrast, equalize, posterize, solarize,
            color, contrast, brightness, sharpness
        ]
        augmentations_without_geo = [
            autocontrast, equalize, posterize, solarize
        ]
        if (aug_list == 'augmentations_without_obj_translation') or (aug_list == 'wotrans'):
            self.aug_list = augmentations_without_obj_translation
        elif aug_list == 'augmentations':
            self.aug_list = augmentations
        elif (aug_list == 'augmentations_all') or (aug_list == 'all'):
            self.aug_list = augmentations_all
        elif aug_list == 'copy':
            self.aug_list = aug_list
        elif (aug_list == 'wogeo') or (aug_list == 'augmentations_without_geo'):
            self.aug_list = augmentations_without_geo
        else: # default = 'augmentations'
            self.aug_list = augmentations


    def __call__(self, results):

        if self.no_jsd:
            img = results['img'].copy()
            results['img'] = self.aug(img)
            return results
        elif self.aug_list == 'copy':
            img = results['img'].copy()
            results['img2'] = img.copy()
            results['img3'] = img.copy()
            results['img_fields'] = ['img', 'img2', 'img3']
            return results
        else:
            img = results['img'].copy()
            results['img_fields'] = ['img']
            for i in range(2, self.num_views+1, 1): # 2, 3, ..., num_views
                results[f'img{i}'] = self.aug(img)
                results['img_fields'].append(f'img{i}')
            return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str

    def aug(self, img):
        ws = np.float32(
            np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        m = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))
        IMAGE_HEIGHT, IMAGE_WIDTH, _ = img.shape
        img_size = (IMAGE_WIDTH, IMAGE_HEIGHT)

        # image_aug = img.copy()
        mix = np.zeros_like(img.copy(), dtype=np.float32)
        for i in range(self.mixture_width):
            image_aug = Image.fromarray(img.copy(), "RGB")
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(self.aug_list)
                outputs = op(image_aug, level=self.aug_severity, img_size=img_size)
                if isinstance(outputs, dict):
                    image_aug = outputs['img']
                elif isinstance(outputs, tuple):
                    images_aug = outputs[0]
                else:
                    images_aug = outputs
            # Preprocessing commutes since all coefficients are convex
            image_aug = np.asarray(image_aug, dtype=np.float32)
            mix += ws[i] * image_aug
        mixed = (1 - m) * img + m * mix
        return mixed
