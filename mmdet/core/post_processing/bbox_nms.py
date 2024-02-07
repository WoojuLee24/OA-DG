# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops.nms import batched_nms

from mmdet.core.bbox.iou_calculators import bbox_overlaps


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   return_inds=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if return_inds:
        return dets, labels[keep], inds[keep]
    else:
        return dets, labels[keep]

# ANALYSIS[CODE=001]: analysis background
def analysis_multiclass_nms(multi_bboxes, multi_scores, score_thr,
                            nms_cfg, max_num=-1, score_factors=None, return_inds=False,
                            img_metas=None):

    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    if img_metas is not None:
        import matplotlib.pyplot as plt
        from thirdparty.dscv.utils.detection_utils import visualize_bbox_xy, get_color_array, pixel2inch
        import torchvision

        corruption = img_metas[0]['corruption']
        severity = img_metas[0]['severity']
        work_dir = img_metas[0]['work_dir']

        img_shape = img_metas[0]['img_shape']
        img = img_metas[0]['img'][0]
        img_norm_cfg = img_metas[0]['img_norm_cfg']

        denormalize = torchvision.transforms.Compose([
            torchvision.transforms.Normalize([0., 0., 0.], 1/img_norm_cfg['std']),
            torchvision.transforms.Normalize(-img_norm_cfg['mean'], [1., 1., 1.])
        ])
        img = denormalize(img).permute(1,2,0)
        img = (img - img.min()) / (img.max() - img.min())
        img = img.cpu().detach().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(pixel2inch(img_shape[1]), pixel2inch(img_shape[0])))
        ax.imshow(img)
        fig.savefig(f'/ws/external/{work_dir}{corruption}{severity}_img.png')
        plt.close(fig)

        def set_title_with_score(multi_scores, i, fig=None, ax=None, fontsize=10):
            title = ''
            CLASSES = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'background']
            for c in range(multi_scores.shape[-1]):
                title += f"{CLASSES[c]}({multi_scores[i, c]:.2f}) "
            fig.suptitle(title, fontsize=fontsize)
        def visualize_for_each_bbox(bboxes, img, img_metas, num_bboxes=50, work_dir='', save_title='each_bboxes', multi_scores=None, fontsize=10):
            img_shape = img_metas[0]['img_shape']

            for i in range(num_bboxes):
                fig, ax = plt.subplots(1, 1, figsize=(pixel2inch(img_shape[1]), pixel2inch(img_shape[0])))
                ax.imshow(img)

                for c in range(num_classes):
                    visualize_bbox_xy(bboxes[i, c, :], fig=fig, ax=ax, color_idx=c, num_colors=num_classes)

                if multi_scores is not None:
                    set_title_with_score(multi_scores, i, fig=fig, ax=ax, fontsize=fontsize)

                ax.axes.set_xlim(0, img_shape[1])
                ax.axes.set_ylim(img_shape[0], 0)
                corruption = img_metas[0]['corruption']
                severity = img_metas[0]['severity']
                fig.savefig(f'/ws/external/{work_dir}{corruption}{severity}_{save_title}{i}.png')
                plt.close(fig)
        def visualize_for_each_class(img, img_metas, bboxes, num_classes=8, work_dir='', score_thr=0.0, save_title='each_classes'):
            img_shape = img_metas[0]['img_shape']

            for c in range(num_classes):
                fig, ax = plt.subplots(1, 1, figsize=(pixel2inch(img_shape[1]), pixel2inch(img_shape[0])))
                ax.imshow(img)

                for i in range(len(multi_scores)):
                    if multi_scores[i, c] > score_thr:
                        visualize_bbox_xy(bboxes[i, c, :], fig=fig, ax=ax, color_idx=c, num_colors=num_classes)

                CLASSES = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'background']
                fig.suptitle(f"{CLASSES[c]}(thr:{score_thr:.2f})")

                ax.axes.set_xlim(0, img_shape[1])
                ax.axes.set_ylim(img_shape[0], 0)
                corruption = img_metas[0]['corruption']
                severity = img_metas[0]['severity']
                fig.savefig(f'/ws/external/{work_dir}{corruption}{severity}_{save_title}_class{c}_thr{score_thr}.png')
                plt.close(fig)

        visualize_for_each_bbox(bboxes, img, img_metas, num_bboxes=len(bboxes), multi_scores=multi_scores, work_dir=work_dir)
        for score_thr in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]:
            visualize_for_each_class(img, img_metas, bboxes, work_dir=work_dir, score_thr=score_thr)

    return 0


def fast_nms(multi_bboxes,
             multi_scores,
             multi_coeffs,
             score_thr,
             iou_thr,
             top_k,
             max_num=-1):
    """Fast NMS in `YOLACT <https://arxiv.org/abs/1904.02689>`_.

    Fast NMS allows already-removed detections to suppress other detections so
    that every instance can be decided to be kept or discarded in parallel,
    which is not possible in traditional NMS. This relaxation allows us to
    implement Fast NMS entirely in standard GPU-accelerated matrix operations.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class+1), where the last column
            contains scores of the background class, but this will be ignored.
        multi_coeffs (Tensor): shape (n, #class*coeffs_dim).
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_thr (float): IoU threshold to be considered as conflicted.
        top_k (int): if there are more than top_k bboxes before NMS,
            only top top_k will be kept.
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept. If -1, keep all the bboxes.
            Default: -1.

    Returns:
        tuple: (dets, labels, coefficients), tensors of shape (k, 5), (k, 1),
            and (k, coeffs_dim). Dets are boxes with scores.
            Labels are 0-based.
    """

    scores = multi_scores[:, :-1].t()  # [#class, n]
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]  # [#class, topk]
    num_classes, num_dets = idx.size()
    boxes = multi_bboxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    coeffs = multi_coeffs[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = bbox_overlaps(boxes, boxes)  # [#class, topk, topk]
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = iou_max <= iou_thr

    # Second thresholding introduces 0.2 mAP gain at negligible time cost
    keep *= scores > score_thr

    # Assign each kept detection to its corresponding class
    classes = torch.arange(
        num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    coeffs = coeffs[keep]
    scores = scores[keep]

    # Only keep the top max_num highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    if max_num > 0:
        idx = idx[:max_num]
        scores = scores[:max_num]

    classes = classes[idx]
    boxes = boxes[idx]
    coeffs = coeffs[idx]

    cls_dets = torch.cat([boxes, scores[:, None]], dim=1)
    return cls_dets, classes, coeffs
