# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time
import copy
import numpy as np

import mmcv
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results
from mmdet.utils.visualize import plot_matrix

def single_gpu_analysis_background(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    work_dir='analysis_background/'):
    model.eval()
    results = []
    dataset = data_loader.dataset
    for i, data in enumerate(data_loader):
        _transform = dataset.pipeline.transforms[1]
        if _transform.__class__.__name__ == 'Corrupt':
            data['img_metas'][0].data[0][0]['corruption'] = _transform.corruption
            data['img_metas'][0].data[0][0]['severity'] = _transform.severity
        else:
            data['img_metas'][0].data[0][0]['corruption'] = 'None'
            data['img_metas'][0].data[0][0]['severity'] = 0

        annotations = [dataset.get_ann_info(i) for i in range(i*data_loader.batch_size, (i+1)*data_loader.batch_size)]
        data['img_metas'][0].data[0][0]['annotations'] = annotations

        data['img_metas'][0].data[0][0]['work_dir'] = work_dir

        with torch.no_grad():
            model(return_loss=False, rescale=True, **data, analysis_background=True)
        break
    return results


def single_gpu_analyze_feature(model,
                    data_loader,
                    orig_dataset=None,
                    show=False,
                    show_dir=None,
                    show_score_thr=0.3,
                    work_dir='analysis_background/',
                    proposal_type='duplicate'):
    model.eval()
    results = []
    dataset = data_loader.dataset
    for i, data in enumerate(data_loader):
        _transform = dataset.pipeline.transforms[1]
        if _transform.__class__.__name__ == 'Corrupt':
            data['img_metas'][0].data[0][0]['corruption'] = _transform.corruption
            data['img_metas'][0].data[0][0]['severity'] = _transform.severity
        else:
            data['img_metas'][0].data[0][0]['corruption'] = 'None'
            data['img_metas'][0].data[0][0]['severity'] = 0

        annotations = [dataset.get_ann_info(i) for i in range(i*data_loader.batch_size, (i+1)*data_loader.batch_size)]
        data['img_metas'][0].data[0][0]['annotations'] = annotations

        data['img_metas'][0].data[0][0]['work_dir'] = show_dir + "/"
        data['img_metas'][0].data[0][0]['proposal_type'] = proposal_type

        if orig_dataset != None: # multi domain test
            orig_data = orig_dataset[i]
            img = orig_data['img'][0].unsqueeze(dim=0)
            data['img'][0] = torch.cat([img, data['img'][0]], dim=0)

        with torch.no_grad():
            model(return_loss=False, rescale=True, **data, analysis='feature')
        break
    return results


def single_gpu_analyze_feature_class(model,
                    data_loader,
                    orig_dataset=None,
                    show=False,
                    show_dir=None,
                    show_score_thr=0.3,
                    work_dir='analysis_background/',
                    proposal_type='duplicate'):

    # to do:
    # test feature on the test mode
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    features_sum = dict()

    for i, data in enumerate(data_loader):
        _transform = dataset.pipeline.transforms[1]
        if _transform.__class__.__name__ == 'Corrupt':
            data['img_metas'][0].data[0][0]['corruption'] = _transform.corruption
            data['img_metas'][0].data[0][0]['severity'] = _transform.severity
        else:
            data['img_metas'][0].data[0][0]['corruption'] = 'None'
            data['img_metas'][0].data[0][0]['severity'] = 0

        annotations = [dataset.get_ann_info(i) for i in range(i*data_loader.batch_size, (i+1)*data_loader.batch_size)]
        data['img_metas'][0].data[0][0]['annotations'] = annotations

        data['img_metas'][0].data[0][0]['work_dir'] = show_dir + "/"
        data['img_metas'][0].data[0][0]['proposal_type'] = proposal_type

        if orig_dataset != None: # multi domain test
            orig_data = orig_dataset[i]
            img = orig_data['img'][0].unsqueeze(dim=0)
            data['img'][0] = torch.cat([img, data['img'][0]], dim=0)

        aug = False
        if aug:
            img = data['img'][0]
            mask = torch.rand_like(img[0:1, 0, :, :]) > 0.8
            mask = mask.float()
            mask = mask.repeat(1, 3, 1, 1)
            img_masked = img * mask  # original image masked
            data['img'][0] = torch.cat([img, img_masked], dim=0) # orig, corr, aug

        with torch.no_grad():
            features = model(return_loss=False, rescale=True, **data, analysis='feature_class')

            if i == 0:
                features_sum = copy.deepcopy(features)


            elif i >= 490:
                for key, value in features.items():
                    if not 'loss' in key:
                        features_sum[key] += value
            else:
                for key, value in features.items():
                    if not 'loss' in key:
                        features_sum[key] += value

            # # debug
            # if i == 5:
            #     break

        for _ in range(batch_size):
            prog_bar.update()

        # get sum of samples with mask
        matrix_sample_number = features_sum['matrix_sample_number(roi_cls)']
        classes = np.shape(matrix_sample_number)[0]
        mask_eye = np.identity(classes, dtype=np.float32)  # [B, B]

        class_matrix = features_sum['matrix_sample_number(roi_cls)']
        class_matrix_same = mask_eye * class_matrix
        class_sum_same = class_matrix_same.sum()
        class_matrix_diff = class_matrix - class_matrix_same
        class_sum_diff = class_matrix_diff.sum() / 2

        for key, value in features_sum.items():
            if 'confusion_matrix' in key:
                features_sum[key] = value / (matrix_sample_number + 1e-6)
                feature_matrix = features_sum[key]
                plt = plot_matrix(feature_matrix, dataset='cityscapes', title=key)
                plt.savefig(f'{show_dir}/{key}.png')
                plt = plot_matrix(feature_matrix, dataset='cityscapes', title=key, normalize='y')
                plt.savefig(f'{show_dir}/{key}_ynorm.png')
            elif 'distance_diff' in key:
                features_sum[key] = value / class_sum_diff
                print('{key}: ', features_sum[key])
            elif 'distance_same' in key:
                features_sum[key] = value / class_sum_same
                print(f'{key}: ', features_sum[key])
            elif 'matrix_sample_number' in key:
                plt = plot_matrix(matrix_sample_number, dataset='cityscapes', title=key, normalize='xy')
                plt.savefig(f'{show_dir}/{key}_xynorm.png')
        break
    return results


def single_gpu_test_feature(model,
                            data_loader,
                            orig_dataset=None,
                            show=False,
                            show_dir=None,
                            show_score_thr=0.3):

    ####
    # [DEV 003]
    # test feature on the single and multi domain
    # forward train mode
    ####
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    batch_size = 1
    features_sum = dict()

    for i, data in enumerate(data_loader):
        orig_data = orig_dataset[i]
        img = orig_data['img'].data
        img = img.unsqueeze(dim=0)
        # img = [orig_data['img'][0].unsqueeze(dim=0)]
        data['img2'] = img

        aug = False
        if aug:
            mask = torch.rand_like(img[:, 0, :, :]) > 0.8
            mask = mask.float()
            mask = mask.repeat(1, 3, 1, 1)
            data['img3'] = img * mask  # original image masked

        with torch.no_grad():
            loss, features = model(return_loss=True, analysis='multi_domain', **data)
            # result = model(return_loss=False, rescale=True, **data)

            if i == 0:
                features_sum = copy.deepcopy(features)

            elif i >= 490:
                for key, value in features.items():
                    if not 'loss' in key:
                        features_sum[key] += value
            else:
                for key, value in features.items():
                    if not 'loss' in key:
                        features_sum[key] += value

        # # debug
        # if i == 5:
        #     break

        for _ in range(batch_size):
            prog_bar.update()

    # get sum of samples with mask
    matrix_sample_number = features_sum['clean_clean_matrix_sample_number']
    classes = np.shape(matrix_sample_number)[0]
    mask_eye = np.identity(classes, dtype=np.float32)  # [B, B]

    class_matrix = features_sum['clean_clean_matrix_sample_number']
    class_matrix_same = mask_eye * class_matrix
    class_sum_same = class_matrix_same.sum()
    class_matrix_diff = class_matrix - class_matrix_same
    class_sum_diff = class_matrix_diff.sum() / 2

    for key, value in features_sum.items():
        if 'confusion_matrix' in key:
            features_sum[key] = value / (matrix_sample_number + 1e-6)
            feature_matrix = features_sum[key]
            plt = plot_matrix(feature_matrix, dataset='cityscapes', title=key)
            plt.savefig(f'{show_dir}/{key}.png')
            plt = plot_matrix(feature_matrix, dataset='cityscapes', title=key, normalize='y')
            plt.savefig(f'{show_dir}/{key}_ynorm.png')
        elif 'distance_diff' in key:
            features_sum[key] = value / class_sum_diff
            print('{key}: ', features_sum[key])
        elif 'distance_same' in key:
            features_sum[key] = value / class_sum_same
            print(f'{key}: ', features_sum[key])
        elif 'matrix_sample_number' in key:
            plt = plot_matrix(matrix_sample_number, dataset='cityscapes', title=key)
            plt.savefig(f'{show_dir}/{key}.png')


    return features_sum


from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
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
        if iou_max < 0.5:
            proposal_gt_labels[total_num_valid:total_num_valid + len_new_bboxes] = 8
        elif 0.5 < iou_min:
            proposal_gt_labels[total_num_valid:total_num_valid + len_new_bboxes] = gt_label
        else:
            raise NotImplementedError

        total_num_valid += len_new_bboxes
        if len(random_bboxes_per_gt) > 0:
            random_bboxes_per_gt = np.stack(random_bboxes_per_gt, axis=0)
            random_bboxes.append(random_bboxes_per_gt)
        continue
    if len(random_bboxes) > 0:
        random_bboxes = np.concatenate(random_bboxes, axis=0)
    else:
        random_bboxes = np.zeros((0, 4))
    return random_bboxes, proposal_gt_labels[:total_num_valid]


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


# def single_gpu_test(model,
#                     data_loader,
#                     show=False,
#                     out_dir=None,
#                     show_score_thr=0.3,
#                     gt_coco=None,
#                     mode='test',
#                     gen_proposal_cfg=None,
#                     num_bboxes=10,
#                     **kwargs):
#     model.eval()
#     results = []
#     dataset = data_loader.dataset
#     prog_bar = mmcv.ProgressBar(len(dataset))
#     break_point = False
#     num_total_proposal = 0
#     total_fn, total_tp = 0, 0
#     proposal_gt_labels_list, pred_labels_list = [], []
#     for i, data in enumerate(data_loader):
#         gt_ann = gt_coco.get_ann_info(i)
#         gt_bboxes = gt_ann['bboxes']
#         img_shape = data['img'][0].shape[2:]
#         if mode == 'test':
#             proposal = torch.zeros((len(gt_bboxes), 5))
#             for j in range(len(gt_bboxes)):  # tl_x, tl_y, br_x, br_y, 0~1
#                 proposal[j, :4] = torch.tensor(gt_bboxes[j])
#                 proposal[j, -1] = 1
#             proposal_list = [proposal]
#         elif mode == 'generate_proposal':
#             gt_labels = gt_ann['labels']
#             label, iou_min, iou_max = gen_proposal_cfg['label'] if gen_proposal_cfg is not None else None, \
#                                       gen_proposal_cfg['iou_min'] if gen_proposal_cfg is not None else 0.9, \
#                                       gen_proposal_cfg['iou_max'] if gen_proposal_cfg is not None else 1.0
#             random_bboxes, proposal_gt_labels = generate_random_bboxes_with_iou(gt_bboxes, iou_min, iou_max, img_shape,
#                                                                                 num_bboxes=num_bboxes,
#                                                                                 gt_labels=gt_labels, label=label,
#                                                                                 max_iter=500, eps=1e-6)
#             ones = np.ones((len(random_bboxes), 1))
#             proposal = np.concatenate([random_bboxes, ones], axis=1)
#             proposal_list = [torch.tensor(proposal, dtype=torch.tensor(gt_bboxes).dtype)]
#             # print(f"proposal len = {len(proposal)}")
#             num_total_proposal += len(proposal)
#             if len(proposal) == 0:
#                 for _ in range(batch_size):
#                     prog_bar.update()
#                     _each_result = np.ndarray((0, 5))
#                     result = [[np.ndarray((0, 5)) for _ in range(len(dataset.CLASSES))]]
#                     results.extend(result)
#                 continue
#         else:
#             raise NotImplementedError
#
#         with torch.no_grad():
#             result = model(return_loss=False, rescale=True, **data, proposals=proposal_list)
#         batch_size = len(result)
#         bbox_results = model.module.roi_head.bbox_results
#         cls_score = bbox_results['cls_score']
#         pred_labels = torch.argmax(cls_score, dim=1)
#
#         try:
#             assert len(pred_labels) == len(proposal)
#         except:
#             print('hello')
#         if gen_proposal_cfg['label'] is not None:
#             fn = torch.sum(pred_labels != gen_proposal_cfg['label'])
#             tp = torch.sum(pred_labels == gen_proposal_cfg['label'])
#             total_fn += fn
#             total_tp += tp
#         else:
#             proposal_gt_labels_list.append(torch.tensor(proposal_gt_labels, device=pred_labels.device))
#             pred_labels_list.append(pred_labels)
#
#         if show or out_dir:
#             if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
#                 img_tensor = data['img'][0]
#             else:
#                 img_tensor = data['img'][0].data[0]
#             img_metas = data['img_metas'][0].data[0]
#             imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
#             assert len(imgs) == len(img_metas)
#
#             for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
#                 h, w, _ = img_meta['img_shape']
#                 img_show = img[:h, :w, :]
#
#                 ori_h, ori_w = img_meta['ori_shape'][:-1]
#                 img_show = mmcv.imresize(img_show, (ori_w, ori_h))
#
#                 if out_dir:
#                     out_file = osp.join(out_dir, img_meta['ori_filename'])
#                 else:
#                     out_file = None
#
#                 model.module.show_result(
#                     img_show,
#                     result[i],
#                     show=show,
#                     out_file=out_file,
#                     score_thr=show_score_thr)
#
#         # encode mask results
#         if isinstance(result[0], tuple):
#             result = [(bbox_results, encode_mask_results(mask_results))
#                       for bbox_results, mask_results in result]
#         results.extend(result)
#
#         for _ in range(batch_size):
#             prog_bar.update()
#
#         if break_point:
#             break
#     if gen_proposal_cfg['label'] is not None:
#         acc = total_tp.item() / num_total_proposal
#     else:
#         total_proposal_gt_labels = torch.cat(proposal_gt_labels_list, dim=0)
#         total_pred_labels = torch.cat(pred_labels_list, dim=0)
#         acc = torch.sum(total_proposal_gt_labels == total_pred_labels).item() / len(total_proposal_gt_labels)
#
#     return results, acc


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
