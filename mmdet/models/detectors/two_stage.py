# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmdet.utils.visualize import visualize_score_distribution, visualize_score_density, visualize_image, get_file_name


@DETECTORS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        # random proposal_list
        if 'random_proposal_cfg' in self.train_cfg.keys():
            kwargs['random_proposal_list'] = self.get_random_proposal_list(img, gt_bboxes, kwargs)

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses


    def get_random_proposal_list(self, img, gt_bboxes, kwargs):
        random_proposal_cfg = self.train_cfg['random_proposal_cfg']
        img_shape = img.shape[2:]
        B = img.shape[0]

        assert 'num_views' in kwargs.keys(), "num_view is required"
        assert random_proposal_cfg['bbox_from'] == 'oagrb', "oagrb is required"
        assert 'multilevel_bboxes' in kwargs.keys() or 'oamix_boxes' in kwargs.keys(), "boxes are required"

        random_proposal_list = []
        device = img.device

        # using boxes from OA-Mix
        if 'multilevel_boxes' in kwargs.keys():
            for _box in kwargs.get('multilevel_boxes', []):
                _box = _box.to(torch.float32).cpu().detach().numpy()
                ious = bbox_overlaps(_box, gt_bboxes[0].cpu().detach().numpy())
                _box = _box[np.max(ious, axis=1) < random_proposal_cfg['iou_max']]
                _box = torch.as_tensor(_box).float().to(device)
                random_proposal_list.append(_box)

        if 'oamix_boxes' in kwargs.keys():
            for i, _box in enumerate(kwargs.get('oamix_boxes', [])):
                _box = _box.to(torch.float32).cpu().detach().numpy()
                ious = bbox_overlaps(_box, gt_bboxes[0].cpu().detach().numpy())
                _box = _box[np.max(ious, axis=1) < random_proposal_cfg['iou_max']]
                _box = torch.as_tensor(_box).float().to(device)
                random_proposal_list[i] = torch.cat([random_proposal_list[i], _box], dim=0)

        # generating new random boxes
        for i in range(B):
            random_bg_bbox = generate_random_bboxes_xy(img_shape,
                                                       num_bboxes=random_proposal_cfg['num_bboxes'],
                                                       bboxes_xy=gt_bboxes[i % kwargs['num_views']].cpu().detach().numpy(),
                                                       scales=random_proposal_cfg['scales'],
                                                       ratios=random_proposal_cfg['ratios'],
                                                       iou_max=random_proposal_cfg['iou_max'],
                                                       iou_min=random_proposal_cfg['iou_min']
                                                       )
            random_bg_bbox = torch.as_tensor(random_bg_bbox[:, :4]).float().to(device)
            random_proposal_list[i] = torch.cat([random_proposal_list[i], random_bg_bbox], dim=0)

        return random_proposal_list

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False, **kwargs):
        """Test without augmentation."""
        debug_cfg = kwargs['debug_cfg'] if 'debug_cfg' in kwargs else None

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        # hook the fpn features
        self.fpn_features = x
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        if debug_cfg:
            if 'given_proposal_list' in debug_cfg:
                if debug_cfg['given_proposal_list']:
                    out_dir = debug_cfg['out_dir']
                    out_dir = out_dir.replace('given', 'augmix.wotrans_plus_rpn.tailv2.1.none_roi.none.none__e2_lw.12')
                    out_dir = out_dir.replace('gaussian_noise/1', 'gaussian_noise/0')
                    out_dir = out_dir.replace('gaussian_noise/2', 'gaussian_noise/0')
                    name = f"{img_metas[0]['ori_filename'].split('.png')[0]}_proposal_list"
                    proposal_list = torch.load(f"{out_dir}/{name}.pt")
            if 'given_proposal_list2' in debug_cfg:
                if debug_cfg['given_proposal_list2']:
                    out_dir = debug_cfg['out_dir']
                    out_dir = out_dir.replace('given2', 'augmix.wotrans_plus_rpn.tailv2.1.none_roi.none.none__e2_lw.12')
                    out_dir = out_dir.replace('gaussian_noise/0', 'gaussian_noise/2')
                    name = f"{img_metas[0]['ori_filename'].split('.png')[0]}_proposal_list"
                    proposal_list = torch.load(f"{out_dir}/{name}.pt")


        bbox_results = self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)

        if debug_cfg:
            if debug_cfg and ('proposal_list' in debug_cfg['save_list']):
                fn = get_file_name(debug_cfg, 'proposal_list', extension='pt', img_meta=img_metas[0])
                torch.save(proposal_list, fn)
            visualize_image(img_meta=img_metas[0], name='original_image', debug_cfg=debug_cfg)
            visualize_score_distribution(proposal_list[0][:, 4], name='proposal_list_score_distribution', bins=50, img_meta=img_metas[0], debug_cfg=debug_cfg)
            visualize_score_density(proposal_list[0], name='proposal_list_score_density', img_meta=img_metas[0], topk=300, debug_cfg=debug_cfg)
            visualize_score_distribution(np.concatenate(bbox_results[0], 0)[:, 4], name='bbox_results_score_distribution', bins=50, img_meta=img_metas[0], debug_cfg=debug_cfg)
            visualize_score_density(bbox_results[0], name='bbox_results_score_density', img_meta=img_metas[0], debug_cfg=debug_cfg)

        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )


def generate_random_bboxes_with_iou(gt_bboxes, iou_min, iou_max, img_shape,
                                    num_bboxes=1000, gt_labels=None, label=None,
                                    max_iter=5000, eps=1e-6):
    h_img, w_img = img_shape
    if gt_labels is not None and label is not None:
        target_mask = gt_labels == label
        if np.sum(target_mask) == 0:
            return np.zeros((0, 4))
        target_gt_bboxes = gt_bboxes[target_mask]
    else:
        target_gt_bboxes = gt_bboxes

    random_bboxes = []
    total_num_valid = 0
    proposal_gt_labels = np.zeros((int(num_bboxes) * len(target_gt_bboxes),))
    for i, (gt_bbox, gt_label) in enumerate(zip(target_gt_bboxes, gt_labels)):
        # gt info
        x1, y1, x2, y2 = gt_bbox
        h_gt, w_gt = y2 - y1, x2 - x1
        area_gt = (x2 - x1) * (y2 - y1)

        # Sample new coordinate (x, y), iou, and ratio
        random_bboxes_per_gt = []
        for j in range(max_iter):
            iou = np.random.uniform(iou_min + eps, iou_max + eps)
            area_overlap_min = iou * area_gt

            area_new_max = area_gt / iou
            area_new = np.random.uniform(area_overlap_min, area_new_max)
            area_overlap = np.random.uniform(area_overlap_min, min(area_gt, area_new))

            h_overlap_min = area_overlap / w_gt
            h_overlap = np.random.uniform(h_overlap_min, h_gt)
            w_overlap = area_overlap / h_overlap
            if not (1 < w_overlap < w_gt):
                continue

            h_new_min = max(np.sqrt(area_gt / (3 * iou)), h_overlap)
            h_new_max = h_overlap * (1 / iou + 1) - area_gt / w_overlap
            h_new = np.random.uniform(h_new_min, h_new_max)
            w_new = area_new / h_new

            x1_new_min = x1 + w_overlap - w_new
            x1_new_max = x2 - w_overlap
            x1_new = np.random.uniform(x1_new_min, x1_new_max)
            x2_new = x1_new + w_new

            y1_new_min = y1 + h_overlap - h_new
            y1_new_max = y2 - h_overlap
            y1_new = np.random.uniform(y1_new_min, y1_new_max)
            y2_new = y1_new + h_new

            bbox = np.stack([x1_new, y1_new, x2_new, y2_new], axis=0)

            _target_gt_bboxes = np.concatenate([target_gt_bboxes[:i], target_gt_bboxes[i+1:]], axis=0)
            all_overlaps = bbox_overlaps(np.expand_dims(bbox, axis=0), _target_gt_bboxes)
            if len(_target_gt_bboxes) == 0 or iou_max < np.max(all_overlaps):
                # print('hello')
                continue

            _area_overlap = (min(x2, x2_new) - max(x1, x1_new)) * (min(y2, y2_new) - max(y1, y1_new))
            _area_new = (x2_new - x1_new) * (y2_new - y1_new)
            _area_union = area_gt + _area_new - _area_overlap
            _iou = _area_overlap / (_area_union + eps)
            if (iou_min < _iou <= iou_max) and \
                    (0 <= x1_new < x2_new <= w_img) and (0 <= y1_new < y2_new <= h_img):
                random_bboxes_per_gt.append(bbox)

            if len(random_bboxes_per_gt) == num_bboxes:
                break

        len_new_bboxes = len(random_bboxes_per_gt)
        # if iou_max < 0.5:
        #     proposal_gt_labels[total_num_valid:total_num_valid + len_new_bboxes] = 8
        # elif 0.5 < iou_min:
        #     proposal_gt_labels[total_num_valid:total_num_valid + len_new_bboxes] = gt_label
        # else:
        #     raise NotImplementedError

        total_num_valid += len_new_bboxes
        if len(random_bboxes_per_gt) > 0:
            random_bboxes_per_gt = np.stack(random_bboxes_per_gt, axis=0)
            random_bboxes.append(random_bboxes_per_gt)
        continue
    if len(random_bboxes) > 0:
        random_bboxes = np.concatenate(random_bboxes, axis=0)
    else:
        random_bboxes = np.zeros((0, 4))
    return random_bboxes    # , proposal_gt_labels[:total_num_valid]


# Random bboxes only augmentation
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
def generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy=None,
                              scales=(0.01, 0.2), ratios=(0.3, 1/0.3),
                              max_iters=500, iou_max=1.0, iou_min=0.0, allow_fg=False, **kwargs):
    # REF: mmdetection/mmdet/datasets/pipelines/transforms.py Cutout
    if isinstance(num_bboxes, tuple) or isinstance(num_bboxes, list):
        num_bboxes = np.random.randint(num_bboxes[0], num_bboxes[1] + 1)
    (img_width, img_height) = img_size

    random_bboxes_xy = np.zeros((num_bboxes, 5))
    total_bboxes = 0
    for i in range(max_iters):
        if total_bboxes >= num_bboxes:
            break
        # Generate a random bbox.
        x1, y1 = np.random.randint(0, img_width), np.random.randint(0, img_height)
        scale = np.random.uniform(*scales) * img_height * img_width
        ratio = np.random.uniform(*ratios)
        bbox_w, bbox_h = int(np.sqrt(scale / ratio)), int(np.sqrt(scale * ratio))
        random_bbox = np.array([[x1, y1, min(x1 + bbox_w, img_width), min(y1 + bbox_h, img_height), 1]])
        if bboxes_xy is not None:
            ious = bbox_overlaps(random_bbox, bboxes_xy)
            if np.max(ious) > iou_max:  # if np.sum(ious) > iou_max:
                continue
            if np.max(ious) < iou_min:
                continue
        random_bboxes_xy[total_bboxes, :] = random_bbox[0]
        total_bboxes += 1
    if total_bboxes != num_bboxes:
        random_bboxes_xy = random_bboxes_xy[:total_bboxes, :]

    return random_bboxes_xy
