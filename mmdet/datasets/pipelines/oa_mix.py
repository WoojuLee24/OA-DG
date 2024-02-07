import warnings

import numpy as np
from PIL import Image

from ..builder import PIPELINES
from .bbox_augmentation import (autocontrast, equalize, posterize, solarize, color, contrast, brightness, sharpness, invert,
                                bboxes_only_rotate, bboxes_only_shear_xy, bboxes_only_translate_xy,
                                bg_only_rotate, bg_only_shear_xy, bg_only_translate_xy)

import cv2
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


def get_aug_list(version):
    if version == 'augmix':
        aug_list = [autocontrast, equalize, posterize, solarize,
                    bboxes_only_rotate, bboxes_only_shear_xy, bboxes_only_translate_xy,
                    bg_only_rotate, bg_only_shear_xy, bg_only_translate_xy,
                    ]
    elif version == 'augmix.all':
        aug_list = [autocontrast, equalize, posterize, solarize, invert,
                    color, contrast, brightness, sharpness,
                    bboxes_only_rotate, bboxes_only_shear_xy, bboxes_only_translate_xy,
                    bg_only_rotate, bg_only_shear_xy, bg_only_translate_xy,
                    ]
    else:
        raise NotImplementedError
    return aug_list


@PIPELINES.register_module()
class OAMix:
    def __init__(self,
                 version='augmix',
                 num_views=2, keep_orig=True, severity=10,
                 mixture_width=3, mixture_depth=-1,   # Mixing strategy (AugMix setting)
                 random_box_scale=(0.01, 0.1), random_box_ratio=(3, 1 / 3),  # multi-level transformation
                 oa_random_box_scale=(0.005, 0.1), oa_random_box_ratio=(3, 1 / 3), num_bboxes=(3, 5), # object-aware mixing
                 spatial_ratio=4, sigma_ratio=0.3,  # Smoothing strategy to improve speed
                 **kwargs):
        super(OAMix, self).__init__()
        self.aug_list = get_aug_list(version)

        self.num_views = num_views  # total number of images to be augmented. the result will be [img1, img2, ..., img{num_views}]
        self.keep_orig = keep_orig  # whether to keep `img1` as the original image or not
        if self.num_views == 1 and self.keep_orig:  # It behaves like identity function.
            warnings.warn('No augmentation will be applied since num_views=1 and keep_orig=True')

        # follow AugMix settings
        self.severity = severity  # strength of transformation (0~10)
        self.aug_prob_coeff = 1.0
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth

        """ Multi-level transformation """
        self.random_box_scale = random_box_scale
        self.random_box_ratio = random_box_ratio

        """ Object-aware mixing """
        self.oa_random_box_scale = oa_random_box_scale
        self.oa_random_box_ratio = oa_random_box_ratio

        self.score_thresh = 10

        # Smoothing strategy (for fg and bg)
        self.spatial_ratio = spatial_ratio
        self.sigma_ratio = sigma_ratio

        """ Etc """
        self._history = {}
        self.kwargs = kwargs

    @staticmethod
    def _get_mask(box, target_shape, spatial_ratio=None, sigma_ratio=None):
        h_img, w_img, c_img = target_shape
        use_blur = (spatial_ratio is not None) and (sigma_ratio is not None)
        if use_blur:
            x1, y1, x2, y2 = np.array(box // spatial_ratio, dtype=np.int32)
            mask = np.zeros((h_img // spatial_ratio, w_img // spatial_ratio, c_img), dtype=np.float32)
        else:
            x1, y1, x2, y2 = box
            mask = np.zeros(target_shape, dtype=np.float32)

        mask[y1:y2, x1:x2, :] = 1.0
        if use_blur:
            sigma_x = (x2 - x1) * sigma_ratio / 3 * 2
            sigma_y = (y2 - y1) * sigma_ratio / 3 * 2
            if not (sigma_x <= 0 or sigma_y <= 0):
                mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)
            mask = cv2.resize(mask, (w_img, h_img))

        return mask

    def get_fg_regions(self, img, gt_bboxes):
        if hasattr(self._history, "fg_box_list"):
            return self._history["fg_box_list"], self._history["fg_mask_list"], self._history["fg_score_list"]
        else:
            fg_box_list, fg_mask_list, fg_score_list = gt_bboxes, [], []
            for i, gt_bbox in enumerate(gt_bboxes):
                """ Object-aware mixing: compute saliency score for each fg region """
                x1, y1, x2, y2 = np.array(gt_bbox, dtype=np.int32)
                if x2 - x1 < self.spatial_ratio or y2 - y1 < self.spatial_ratio:
                    # If it is too small, the score will be -1.
                    fg_score_list.append(-1)
                else:
                    bbox_img = img[y1:y2, x1:x2]
                    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
                    (success, saliency_map) = saliency.computeSaliency(bbox_img)
                    saliency_score = np.mean((saliency_map * 255).astype("uint8"))
                    fg_score_list.append(saliency_score)

                # Hacks to speed up blurred mask generation
                fg_mask = self._get_mask(gt_bbox, img.shape, spatial_ratio=self.spatial_ratio, sigma_ratio=self.sigma_ratio)
                fg_mask_list.append(fg_mask)

            self._history.update({"fg_box_list": fg_box_list})
            self._history.update({"fg_mask_list": fg_mask_list})
            self._history.update({"fg_score_list": fg_score_list})
            return fg_box_list, fg_mask_list, fg_score_list

    def get_random_regions(self, img, scale, ratio,
                           num_bboxes=None, use_blur=False,
                           return_score=False, fg_box_list=None, fg_score_list=None,
                           max_iters=50, eps=1e-6):
        if return_score:
            assert fg_box_list is not None and fg_score_list is not None
        (h_img, w_img, c_img) = img.shape

        random_box_list, random_mask_list, random_score_list = [], [], []

        # Randomly determines the number of boxes to be generated
        target_num_bboxes = np.random.randint(*num_bboxes) if isinstance(num_bboxes, tuple) else num_bboxes
        for i in range(max_iters):
            # Stop random region generation when the determined number of boxes has been generated.
            if len(random_mask_list) >= target_num_bboxes:
                break

            # Generate a random bbox.
            x1, y1 = np.random.randint(0, w_img), np.random.randint(0, h_img)
            _scale = np.random.uniform(*scale) * h_img * w_img
            _ratio = np.random.uniform(*ratio)
            bbox_w, bbox_h = int(np.sqrt(_scale / _ratio)), int(np.sqrt(_scale * _ratio))

            if x1 + bbox_w > w_img or y1 + bbox_h > h_img:
                continue # incorrectly generated box

            x2, y2 = min(x1 + bbox_w, w_img), min(y1 + bbox_h, h_img)
            random_box = np.array([[x1, y1, x2, y2]])

            # Except if it overlaps with existing boxes
            ious = bbox_overlaps(random_box, np.asarray(random_box_list))
            if np.sum(ious) > eps:
                continue

            # compute saliency score for each fg region
            if return_score:
                ious = bbox_overlaps(random_box, fg_box_list)

                final_score = float("inf")
                if np.sum(ious) > eps:
                    # If the random box is overlapped with fg region
                    for i, (iou, fg_box, fg_score) in enumerate(zip(ious[0], fg_box_list, fg_score_list)):
                        # If there is no overlapping area, it is not reflected in the saliency score.
                        x1_fg, y1_fg, x2_fg, y2_fg = fg_box
                        if iou == 0.0 or x2_fg - x1_fg < 1 or y2_fg - y1_fg < 1:
                            continue
                        if fg_score < final_score:
                            final_score = fg_score
                random_score_list.append(final_score)

            # Generate the mask of the random bbox.
            if use_blur:
                random_mask = self._get_mask(
                    random_box[0], img.shape, spatial_ratio=self.spatial_ratio, sigma_ratio=self.sigma_ratio)
            else:
                random_mask = self._get_mask(random_box[0], img.shape)
            random_mask_list.append(random_mask)
            random_box_list += list(random_box)

        if return_score:
            return random_box_list, random_mask_list, random_score_list
        else:
            return random_box_list, random_mask_list


    def __call__(self, results, *args, **kwargs):
        results['custom_field'] = []
        for i in range(1, self.num_views + 1):
            if i == 1:
                self._history = {}
                if not self.keep_orig:
                    results['img'] = self.oamix(results['img'].copy(), results['gt_bboxes'].copy())
                results['img_fields'] = ['img']
            else:
                results[f'img{i}'] = self.oamix(results['img'].copy(), results['gt_bboxes'].copy())
                results['img_fields'] += [f'img{i}']
                results[f'gt_bboxes{i}'] = results[f'gt_bboxes'].copy()
                results[f'oamix_boxes'] = np.stack(self._history["oa_random_box_list"], axis=0)  # oamix_boxes: only bg
                results['custom_field'] += [f'img{i}', f'gt_bboxes{i}', f'oamix_boxes']
                results[f'multilevel_boxes'] = self._history["random_box_list"]  # multilevel_boxes: random, i.e. fg + bg.
                results['custom_field'] += [f'multilevel_boxes']

        return results


    def oamix(self, img, gt_bboxes):
        img = np.asarray(img, dtype=np.uint8)
        h_img, w_img, _ = img.shape
        img_size = (w_img, h_img)

        ws = np.float32(np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))

        # Multi-level transformation: get_random_regions() & get_masks()
        random_box_list, random_mask_list = self.get_random_regions(
            img, self.random_box_scale, self.random_box_ratio, num_bboxes=(1, 3))
        self._history.update({"random_box_list": np.stack(random_box_list, axis=0)})
        fg_box_list, fg_mask_list, fg_score_list = self.get_fg_regions(img=img, gt_bboxes=gt_bboxes)

        # Initialize I_oamix with zeros
        img_oamix = np.zeros_like(img.copy(), dtype=np.float32)
        for i in range(self.mixture_width):
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            img_aug = Image.fromarray(img.copy(), "RGB")
            for _ in range(depth):
                """ Multi-level transformation """
                # Initialize I_aug with zeros
                img_tmp = np.zeros_like(img, dtype=np.float32)
                for _randbox, _randmask in zip(random_box_list, random_mask_list):
                    img_tmp += _randmask * self.aug(img_aug, img_size, fg_box_list, fg_mask_list)

                union_mask = np.max(random_mask_list, axis=0)
                img_aug = np.asarray(
                    img_tmp + (1.0 - union_mask) * self.aug(img_aug, img_size, fg_box_list, fg_mask_list), dtype=np.uint8)

            img_oamix += ws[i] * np.asarray(img_aug, dtype=np.float32)

        """ Object-aware mixing """
        oa_target_box_list, oa_target_mask_list, oa_target_score_list = self.get_regions_for_object_aware_mixing(
            img, fg_box_list, fg_mask_list, fg_score_list)
        img_oamix = self.object_aware_mixing(img, img_oamix, oa_target_mask_list, oa_target_score_list)

        return np.asarray(img_oamix, dtype=np.uint8)

    def get_regions_for_object_aware_mixing(self, img, fg_box_list, fg_mask_list, fg_score_list):
        oa_target_box_list, oa_target_mask_list, oa_target_score_list = [], [], []
        for idx, (box, mask, score) in enumerate(zip(fg_box_list, fg_mask_list, fg_score_list)):
            # For the objects with low saliency score,
            if score <= self.score_thresh:
                oa_target_box_list.append(box)
                oa_target_mask_list.append(mask)
                oa_target_score_list.append(score)
        oa_random_box_list, oa_random_mask_list, oa_random_score_list = self.get_random_regions(
            img, self.oa_random_box_scale, self.oa_random_box_ratio,
            num_bboxes=min(max(len(oa_target_box_list), 1), 5),
            return_score=True, fg_box_list=fg_box_list, fg_score_list=fg_score_list
        )
        oa_target_box_list += oa_random_box_list
        oa_target_mask_list += oa_random_mask_list
        oa_target_score_list += oa_random_score_list
        self._history.update({"oa_random_box_list": oa_random_box_list})
        return oa_target_box_list, oa_target_mask_list, oa_target_score_list

    def aug(self, img, img_size, fg_box_list, fg_mask_list):
        op = np.random.choice(self.aug_list)
        if op in [autocontrast, equalize, posterize, solarize, color, contrast, brightness, sharpness]:
            if type(img) == np.ndarray:
                img = Image.fromarray(img, "RGB")
            pil_img = op(img, level=self.severity, img_size=img_size)
        elif op in [invert]:
            if not isinstance(img, np.ndarray):
                img = np.asarray(img, dtype=np.uint8)
            tx = 1 if np.random.random() > 0.5 else -1
            ty = 1 if np.random.random() > 0.5 else -1
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            pil_img = - cv2.warpAffine(img, M, (0, 0))
        else:
            pil_img = op(img, level=self.severity, img_size=img_size, bboxes_xy=fg_box_list, mask_bboxes=fg_mask_list, fillmode='oa')
        return pil_img

    def object_aware_mixing(self, img, img_aug, mask_list, score_list):
        m = np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff)

        orig, aug = np.zeros_like(img, dtype=np.float32), np.zeros_like(img, dtype=np.float32)
        mask_sum = np.zeros_like(img, dtype=np.float32)
        mask_max_list = []
        for i, (mask, score) in enumerate(zip(mask_list, score_list)):
            # Get union of masks
            mask_sum += mask
            mask_max_list.append(mask)
            mask_max = np.max(mask_max_list, axis=0)
            mask_overlap = mask_sum - mask_max

            # For the objects with low saliency score, m ~ U(0.0, 0.5)
            if score <= self.score_thresh:
                m_oa = np.float32(np.random.uniform(0.0, 0.5))
            else:
                m_oa = np.float32(np.random.uniform(0.0, 1.0))
            orig += (1.0 - m_oa) * img * (mask - mask_overlap * 0.5)
            aug += m_oa * img_aug * (mask - mask_overlap * 0.5)
            mask_sum = mask_max

        img_oamix = orig + aug

        img_oamix += (1.0 - m) * img * (1.0 - mask_sum)
        img_oamix += m * img_aug * (1.0 - mask_sum)
        img_oamix = np.clip(img_oamix, 0, 255)

        return img_oamix

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str