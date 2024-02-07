""" Parsing the result from text to dictionary.

Example:
    python parse_txt2dict.py ${test_robustness_result.txt} ${config_file.py}
Functions:
    get_dictionary(file_path) returns the following dictionary:
        {
            'gaussian_noise': {
                'severity0': {
                    'time_loading_annotations_into_memory': '0.10',
                    'time_loading_and_preparing_results': '0.16',
                    'time_evaluate_annotation_type_bbox': '6.56',
                    'time_accumulating_evaluation_results': '0.30',dsms
                    'average_precision': {
                        'IoU=0.50:0.95|area=all|maxDets=100': '0.409',
                        'IoU=0.50|area=all|maxDets=100': '0.676',
                        'IoU=0.75|area=all|maxDets=100': '0.417',
                        'IoU=0.50:0.95|area=small|maxDets=100': '0.191',
                        'IoU=0.50:0.95|area=medium|maxDets=100': '0.408',
                        'IoU=0.50:0.95|area=large|maxDets=100': '0.592'},
                    'average_recall': {
                        'IoU=0.50:0.95|area=all|maxDets=1': '0.262',
                        'IoU=0.50:0.95|area=all|maxDets=10': '0.461',
                        'IoU=0.50:0.95|area=all|maxDets=100': '0.493',
                        'IoU=0.50:0.95|area=small|maxDets=100': '0.251',
                        'IoU=0.50:0.95|area=medium|maxDets=100': '0.489',
                        'IoU=0.50:0.95|area=large|maxDets=100': '0.687'}},
                ...
                'severity1': { ... }
            'mPC': {
                'average_precision': { ... }
                'average_recall': { ... }
            }
        }

    get_minimal_dictionary(dictionary) returns a dictionary with the following key values corresponding to one value:
        'cleanP_all', 'cleanP_small', 'cleanP_medium', 'cleanP_large'
        , 'corr_mPC_all', 'corr_mPC_small', 'corr_mPC_medium', 'corr_mPC_large'
        , 'gaussian_noise', 'shot_noise', 'impulse_noise'
        , 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'
        , 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
"""

import sys
import re


def get_minimal_dictionary(dictionary):
    minimal_dictionary = {}
    keys = ['cleanP_all', 'cleanP_small', 'cleanP_medium', 'cleanP_large', 'corr_mPC_all', 'corr_mPC_small',
            'corr_mPC_medium', 'corr_mPC_large'
        , 'gaussian_noise', 'shot_noise', 'impulse_noise'
        , 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'
        , 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    for key in keys:
        minimal_dictionary[key] = -1.0

    for key_corruption_type in dictionary.keys():

        ''' corr_mPC_ '''
        if key_corruption_type == 'mPC':
            for area in ['all', 'small', 'medium', 'large']:
                minimal_dictionary['corr_mPC_' + area] = float(
                    dictionary[key_corruption_type]['average_precision']['IoU=0.50:0.95|area=' + area + '|maxDets=100'])
            continue

        ''' cleanP_ '''
        if 'severity0' in dictionary[key_corruption_type]:
            for area in ['all', 'small', 'medium', 'large']:
                minimal_dictionary['cleanP_' + area] = float(
                    dictionary[key_corruption_type]['severity0']['average_precision'][
                        'IoU=0.50:0.95|area=' + area + '|maxDets=100'])

        ''' corruptions '''
        mean_score = 0
        for i in [1, 2, 3, 4, 5]:
            score = float(dictionary[key_corruption_type]['severity' + str(i)]['average_precision'][
                              'IoU=0.50:0.95|area=all|maxDets=100']) if 'average_precision' in dictionary[key_corruption_type]['severity' + str(i)]\
                else float(0.0)
            mean_score = mean_score + score
        minimal_dictionary[key_corruption_type] = float(mean_score / 5)

    for key in minimal_dictionary.keys():
        print('key:', key, ' value:', minimal_dictionary[key] * 100)

    return minimal_dictionary


def get_dictionary(file_path):

    dictionary = {}

    mpc = False
    with open(file_path) as file:
        for line in file:
            '''Corruption Type & Severity'''
            if line.startswith('Testing '):  # e.g., "Testing gaussian_noise at severity 0"
                corruption_type = line.split()[1]
                severity = int(line.split()[4])
                if not corruption_type in dictionary:
                    dictionary[corruption_type] = {}
                if not "severity" + str(severity) in dictionary[corruption_type]:
                    dictionary[corruption_type]["severity" + str(severity)] = {}

            '''time'''
            if line.startswith('loading annotations into memory...'):
                time_loading_annotations_into_memory = re.split('[=,s]', file.readline())[1]
                dictionary[corruption_type]["severity" + str(severity)][
                    "time_loading_annotations_into_memory"] = time_loading_annotations_into_memory
            if line.startswith('Loading and preparing results...'):
                time_loading_and_preparing_results = re.split('[=,s]', file.readline())[1]
                dictionary[corruption_type]["severity" + str(severity)][
                    "time_loading_and_preparing_results"] = time_loading_and_preparing_results
            if line.startswith('Evaluate annotation type *bbox*'):
                time_evaluate_annotation_type_bbox = re.split('[=,s]', file.readline())[1]
                dictionary[corruption_type]["severity" + str(severity)][
                    "time_evaluate_annotation_type_bbox"] = time_evaluate_annotation_type_bbox
            if line.startswith('Accumulating evaluation results...'):
                time_accumulating_evaluation_results = re.split('[=,s]', file.readline())[1]
                dictionary[corruption_type]["severity" + str(severity)][
                    "time_accumulating_evaluation_results"] = time_accumulating_evaluation_results

            '''Average Precision & Average Recall & Mean Performance under Corruption [mPC] (bbox)'''
            if line.startswith('Mean Performance under Corruption [mPC] (bbox)'):
                mpc = True
                dictionary['mPC'] = {}

            if line.startswith(' Average Precision'):
                score_type = 'average_precision'
                if mpc:
                    dictionary['mPC'][score_type] = {}
                else:
                    dictionary[corruption_type]["severity" + str(severity)][score_type] = {}

                words = re.split('[=,|,\[,\],]', line.replace(' ', '').replace('\n',
                                                                               ''))  # e.g., " Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.409"
                while words[0].startswith('AveragePrecision'):
                    iou = words[2]
                    area = words[4]
                    max_dets = words[6]
                    score = words[8]
                    if mpc:
                        dictionary['mPC'][score_type]['IoU=' + iou + '|area=' + area + '|maxDets=' + max_dets] = score
                    else:
                        dictionary[corruption_type]["severity" + str(severity)][score_type][
                            'IoU=' + iou + '|area=' + area + '|maxDets=' + max_dets] = score
                    words = re.split('[=,|,\[,\],]', file.readline().replace(' ', '').replace('\n', ''))

                score_type = 'average_recall'
                if mpc:
                    dictionary['mPC'][score_type] = {}
                else:
                    dictionary[corruption_type]["severity" + str(severity)][score_type] = {}

                while words[0].startswith('AverageRecall'):
                    iou = words[2]
                    area = words[4]
                    max_dets = words[6]
                    score = words[8]
                    if mpc:
                        dictionary['mPC'][score_type]['IoU=' + iou + '|area=' + area + '|maxDets=' + max_dets] = score
                    else:
                        dictionary[corruption_type]["severity" + str(severity)][score_type][
                            'IoU=' + iou + '|area=' + area + '|maxDets=' + max_dets] = score
                    words = re.split('[=,|,\[,\],]', file.readline().replace(' ', '').replace('\n', ''))
    return dictionary

import mmcv
from mmcv import Config

def print_config_information(file_path):
    def print_dict(dict, name):
        print(f'  - {name}: (', end='')
        for key, value in dict.items():
            print(f'{key}={value}, ', end='')
        print(f')')
    cfg = Config.fromfile(file_path)

    ''' Model'''
    model = cfg.model
    print(f'=== config information ===')
    print(f'[model]')
    print_dict(model.rpn_head.loss_cls, 'rpn_cls')
    print_dict(model.rpn_head.loss_bbox, 'rpn_bbox')
    print_dict(model.roi_head.bbox_head.loss_cls, 'roi_cls')
    print_dict(model.roi_head.bbox_head.loss_bbox, 'roi_bbox')

    print(f'[data]')
    data = cfg.data
    print(f'  - samples_per_gpu={data.samples_per_gpu}, workers_per_gpu={data.workers_per_gpu}')
    train_pipeline = data.train.dataset.pipeline
    for i in range(len(train_pipeline)):
        if train_pipeline[i].type == 'AugMix':
            print_dict(train_pipeline[i], f'data.train.dataset.pipeline[{i}]')

    print(f'[runtime]')
    print_dict(cfg.evaluation, 'evaluation')
    print_dict(cfg.optimizer, 'optimizer')
    print(f'==========================')

def main():
    txt_file_path = sys.argv[1]
    config_file_path = sys.argv[2]
    if len(sys.argv) < 2:
        print("Insufficient arguments")
        sys.exit()
    print('txt file path : ' + txt_file_path)
    print('config file path : ' + config_file_path)

    dictionary = get_dictionary(txt_file_path)
    minimal_dictionary = get_minimal_dictionary(dictionary)
    for key in minimal_dictionary.keys():
        print('key:', key, ' value:', minimal_dictionary[key] * 100)

if __name__ == '__main__':
    main()
