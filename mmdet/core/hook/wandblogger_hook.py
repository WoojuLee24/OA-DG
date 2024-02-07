# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.runner import HOOKS
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.checkpoint import CheckpointHook
from mmcv.runner.hooks.logger.wandb import WandbLoggerHook

from mmdet.core import DistEvalHook, EvalHook

import pdb
import matplotlib.pyplot as plt
import numpy as np
import torch

@HOOKS.register_module()
class WandbLogger(WandbLoggerHook):
    """WandbLogger logs metrics, saves model checkpoints as W&B Artifact, and
    logs model prediction as interactive W&B Tables.

    - Metrics: The WandbLogger will automatically log training
        and validation metrics.

    - Checkpointing: If `log_checkpoint` is True, the checkpoint saved at
        every checkpoint interval will be saved as W&B Artifacts.
        Please refer to https://docs.wandb.ai/guides/artifacts/model-versioning
        to learn more about model versioning with W&B Artifacts.
        Note: This depends on the `CheckpointHook` whose priority is more
        than `WandbLogger`.

    - Checkpoint Metadata: If `log_checkpoint_metadata` is True, every
        checkpoint artifact will have a metadata associated with it.
        The metadata contains the evaluation metrics computed on validation
        data with that checkpoint along with the current epoch. If True, it
        also marks the checkpoint version with the best evaluation metric with
        a 'best' alias. You can choose the best checkpoint in the W&B Artifacts
        UI using this.
        Note: It depends on `EvalHook` whose priority is more than WandbLogger.

    - Evaluation: At every evaluation interval, the `WandbLogger` logs the
        model prediction as interactive W&B Tables. The number of samples
        logged is given by `num_eval_images`. Please refer to
        https://docs.wandb.ai/guides/data-vis to learn more about W&B Tables.
        Currently, the `WandbLogger` logs the predicted bounding boxes along
        with the ground truth at every evaluation interval.
        Note: This depends on the `EvalHook` whose priority is more than
        `WandbLogger`. Also note that the data is just logged once and
        subsequent evaluation tables uses reference to the logged data to save
        memory usage.

    ```
    Example:
        log_config = dict(
            interval=10,
            hooks=[
                dict(type='WandbLogger',
                     wandb_init_kwargs={
                         'entity': WANDB_ENTITY,
                         'project': WANDB_PROJECT_NAME
                     },
                     logging_interval=10,
                     log_checkpoint=True,
                     log_checkpoint_metadata=True,
                     num_eval_images=100)
            ])
    ```

    Args:
        wandb_init_kwargs (dict): A dict passed to wandb.init to initialize
            a W&B run. Please refer to https://docs.wandb.ai/ref/python/init
            for possible key-value pairs.
        logging_interval (int): Logging interval (every k iterations).
            Default 10.
        log_checkpoint (bool): Save the checkpoint at every checkpoint interval
            as W&B Artifacts. Use this for model versioning where each version
            is a checkpoint.
            Default: False
        log_checkpoint_metadata (bool): Log the evaluation metrics computed
            on the validation data with the checkpoint, along with current
            epoch as a metadata to that checkpoint.
            Default: True
        num_eval_images (int): Number of validation images to be logged.
            Default: 100
    """

    def __init__(self,
                 wandb_init_kwargs=None,
                 interval=10,
                 log_map_every_iter=True,
                 log_checkpoint=False,
                 log_checkpoint_metadata=False,
                 num_eval_images=100,
                 **kwargs):
        super(WandbLogger, self).__init__(wandb_init_kwargs, interval,
                                          **kwargs)

        self.log_checkpoint = log_checkpoint
        self.log_checkpoint_metadata = log_checkpoint_metadata
        self.num_eval_images = num_eval_images
        self.log_eval_metrics = True
        self.log_map_every_iter = log_map_every_iter
        self.best_score = 0
        self.val_step = 0
        self.best_mAP = 0

        self.val_dataloaders = None

    @master_only
    def before_run(self, runner):
        super(WandbLogger, self).before_run(runner)
        self.cfg = self.wandb.config
        # Check if configuration is passed to wandb.
        if len(dict(self.cfg)) == 0:
            warnings.warn(
                'To log mmdetection Config, '
                'pass it to init_kwargs of WandbLogger.', UserWarning)

        # Check if EvalHook and CheckpointHook are available.
        for hook in runner.hooks:
            if isinstance(hook, CheckpointHook):
                self.ckpt_hook = hook
            if isinstance(hook, (EvalHook, DistEvalHook)):
                self.eval_hook = hook

        # If CheckpointHook is not available turn off log_checkpoint.
        try:
            self._check_priority(self.ckpt_hook)
        except AttributeError:
            self.log_checkpoint = False
            warnings.warn('To use log_checkpoint turn use '
                          'CheckpointHook.', UserWarning)

        # If EvalHook/DistEvalHook is not present set
        # num_eval_images to zero.
        try:
            self.val_dataloader = self.eval_hook.dataloader
            self.val_dataset = self.val_dataloader.dataset
            self._check_priority(self.eval_hook)
        except AttributeError:
            self.num_eval_images = 0
            self.log_checkpoint_metadata = False
            self.log_eval_metrics = False
            warnings.warn(
                'To log num_eval_images turn validate '
                'to True in train_detector.', UserWarning)

        from mmdet.datasets import (build_dataloader, build_dataset)
        if hasattr(runner, 'wandb_cfg'):
            cfg = runner.wandb_cfg
            if hasattr(cfg.data, 'vals'):
                vals_cfg = cfg.data.vals
                val_datasets = [build_dataset(_val_cfg, dict(test_mode=True)) for _val_cfg in vals_cfg]
                self.val_dataloaders = [
                    build_dataloader(
                        ds,
                        samples_per_gpu=1,
                        workers_per_gpu=cfg.data.workers_per_gpu,
                        dist=False,
                        shuffle=False)
                    for ds in val_datasets
                ]
                self.best_mAP_dict = dict()
                for _dataloader in self.val_dataloaders:
                    name = _dataloader.dataset.ann_file.split('/ws/data/S-DGOD/')[1].split('/')[0].lower()
                    _dataloader.name = name
                    self.best_mAP_dict[name] = 0

        # If num_eval_images is greater than zero, create
        # and log W&B table for validation data.
        if self.num_eval_images > 0:
            # Initialize data table
            self._init_data_table()
            # Add data to the table
            self._add_ground_truth()
            # Log ground truth data
            self._log_data_table()

        # Define a custom x-axes for validation metrics.
        if self.log_eval_metrics:
            self.wandb.define_metric('val/val_step')
            self.wandb.define_metric('val/*', step_metric='val/val_step')

    @master_only
    def after_train_epoch(self, runner):
        from mmdet.apis.test import single_gpu_test
        if not self.log_map_every_iter:
            if self.log_eval_metrics:
                if self.eval_hook.by_epoch:
                    if self.every_n_epochs(
                            runner,
                            self.eval_hook.interval) or self.is_last_epoch(runner):
                        results = single_gpu_test(runner.model, self.val_dataloader, show=False)
                        eval_results = self.val_dataset.evaluate(results, logger='silent')
                        # results = self.eval_hook.results
                        # eval_results = self.val_dataset.evaluate(
                        #     results, logger='silent')
                        for key, val in eval_results.items():
                            if isinstance(val, str):
                                continue
                            self.wandb.log({f'val/{key}': val}, commit=False)
                        self.wandb.log({'val/val_step': self.val_step})
                        self.val_step += 1

            if self.log_checkpoint:
                if self.ckpt_hook.by_epoch:
                    if self.every_n_epochs(runner, self.ckpt_hook.interval) or (
                            self.ckpt_hook.save_last
                            and self.is_last_epoch(runner)):
                        if self.log_checkpoint_metadata and self.eval_hook:
                            metadata = self._get_ckpt_metadata(runner)
                            aliases = [f'epoch_{runner.epoch+1}', 'latest']
                            if self._is_best_ckpt(metadata):
                                aliases.append('best')
                            self._log_ckpt_as_artifact(self.ckpt_hook.out_dir,
                                                       runner.epoch, aliases,
                                                       metadata)
                        else:
                            aliases = [f'epoch_{runner.epoch+1}', 'latest']
                            self._log_ckpt_as_artifact(self.ckpt_hook.out_dir,
                                                       runner.epoch, aliases)

            if self.num_eval_images > 0:
                if self.eval_hook.by_epoch:
                    if self.every_n_epochs(runner, self.eval_hook.interval) or self.is_last_epoch(runner):
                        # results = self.eval_hook.results
                        results = self.eval_hook.latest_results
                        # Initialize evaluation table
                        self._init_pred_table()
                        # Log predictions
                        self._log_predictions(results, runner.epoch + 1)
                        # Log the table
                        self._log_eval_table()

    def after_train_iter(self, runner):
        from mmdet.apis.test import single_gpu_test
        super(WandbLogger, self).after_train_iter(runner)
        if 'yolo' in self.init_kwargs['name']:
            for wandb_feature, value in runner.model.module.wandb_features.items():
                self.wandb.log({'train/' + wandb_feature: value})
            runner.model.module.wandb_features.clear()
        elif self.every_n_iters(runner, self.interval):
            # save the wandb_features
            self.log_wandb_feature(runner, split='train/')
            if self.log_map_every_iter:
                # save the feature maps
                self.log_feature_map(runner)

                # measure mAP and save the results on the validation dataset
                if self.val_dataloaders is not None:
                    for val_dataloader in self.val_dataloaders:
                        results = single_gpu_test(runner.model, val_dataloader, show=False)
                        eval_results = val_dataloader.dataset.evaluate(results, logger='silent')
                        print(f"eval_results for {val_dataloader.name}: ", eval_results)

                        if val_dataloader.name == 'daytime_sunny':
                            for key, value in eval_results.items():
                                self.wandb.log({"val/" + f"{key}": value})
                        for key, value in eval_results.items():
                            self.wandb.log({"val/" + f"{key}_{val_dataloader.name}": value})

                        if 'bbox_mAP' in eval_results.keys():
                            if val_dataloader.name == 'daytime_sunny':
                                if eval_results['bbox_mAP'] > self.best_mAP:
                                    runner.save_checkpoint(runner.work_dir, f'best_iter.pth')
                                    self.best_mAP = eval_results['bbox_mAP']
                            if eval_results['bbox_mAP'] > self.best_mAP_dict[val_dataloader.name]:
                                print(f"best mAP: {self.best_mAP_dict[val_dataloader.name]} to {eval_results['bbox_mAP']}")
                                runner.save_checkpoint(runner.work_dir, f'best_iter_{val_dataloader.name}.pth')
                                self.best_mAP_dict[val_dataloader.name] = eval_results['bbox_mAP']

                        elif 'mAP' in eval_results.keys():
                            if val_dataloader.name == 'daytime_sunny':
                                if eval_results['mAP'] > self.best_mAP:
                                    runner.save_checkpoint(runner.work_dir, f'best_iter.pth')
                                    self.best_mAP = eval_results['mAP']
                            if eval_results['mAP'] > self.best_mAP_dict[val_dataloader.name]:
                                print(f"best mAP for {val_dataloader.name}: {self.best_mAP_dict[val_dataloader.name]} to {eval_results['mAP']}")
                                runner.save_checkpoint(runner.work_dir, f'best_iter_{val_dataloader.name}.pth')
                                self.best_mAP_dict[val_dataloader.name] = eval_results['mAP']
                else:
                    results = single_gpu_test(runner.model, self.val_dataloader, show=False)
                    eval_results = self.val_dataset.evaluate(results, logger='silent')
                    print("eval_results: ", eval_results)

                    for key, value in eval_results.items():
                        self.wandb.log({"val/" + key: value})

                    if 'bbox_mAP' in eval_results.keys():
                        if eval_results['bbox_mAP'] > self.best_mAP:
                            print(f"best mAP: {self.best_mAP} to {eval_results['bbox_mAP']}")
                            runner.save_checkpoint(runner.work_dir, 'best_iter.pth')
                            self.best_mAP = eval_results['bbox_mAP']

                    elif 'mAP' in eval_results.keys():
                        if eval_results['mAP'] > self.best_mAP:
                            print(f"best mAP: {self.best_mAP} to {eval_results['mAP']}")
                            runner.save_checkpoint(runner.work_dir, 'best_iter.pth')
                            self.best_mAP = eval_results['mAP']

                # # ce_loss and jsd_loss for validation is not possible now.
                # self.log_wandb_feature(runner, split='val/')
                # # Initialize evaluation table
                # self._init_pred_table()
                # # Log predictions
                # self._log_predictions(results, runner.iter + 1)
                # # Log the table
                # self._log_eval_table()

    def log_feature_map(self, runner):
        # save the feature maps
        if 'wandb' in runner.model.module.train_cfg:
            if 'log' in runner.model.module.train_cfg.wandb:
                if 'features_list' in runner.model.module.train_cfg.wandb.log:
                    # save the rpn feature maps
                    if 'rpn_head.rpn_cls' in runner.model.module.train_cfg.wandb.log.features_list:
                        data = runner.model.module.wandb_data
                        plt = runner.model.module.save_the_result_img(data)
                        self.wandb.log({
                            f"rpn_head.rpn_cls.feature_map": self.wandb.Image(plt)
                        })
                        plt.close()
                    # save the fpn feature maps
                    if ('neck.fpn_convs.0.conv' in runner.model.module.train_cfg.wandb.log.features_list) \
                            or ('neck.fpn_convs.1.conv' in runner.model.module.train_cfg.wandb.log.features_list) \
                            or ('neck.fpn_convs.2.conv' in runner.model.module.train_cfg.wandb.log.features_list) \
                            or ('neck.fpn_convs.3.conv' in runner.model.module.train_cfg.wandb.log.features_list):
                        plt = runner.model.module.save_the_fpn_img()
                        self.wandb.log({
                            f"neck.fpn_convs.feature_map": self.wandb.Image(plt)
                        })
                        plt.close()

    def log_wandb_feature(self, runner, split="train/"):
        # save the loss and jsd loss log
        for wandb_feature, value in runner.model.module.wandb_features.items():
            self.wandb.log({split + wandb_feature: value})
        runner.model.module.wandb_features.clear()


        if hasattr(runner.model.module.rpn_head.loss_cls, 'wandb_features'):
            # save the loss and jsd loss of rpn_cls. rpn_cls consists of 5 feature maps.
            for wandb_feature, value in runner.model.module.rpn_head.loss_cls.wandb_features.items():
                if isinstance(value, list):
                    for i, v in enumerate(value):
                        self.wandb.log({split + wandb_feature + '_layer' + str(i): v})
                else:
                    self.wandb.log({split + wandb_feature: value})
            loss_module = runner.model.module.rpn_head.loss_cls
            loss_module.wandb_features[f'ce_loss({loss_module.wandb_name})'].clear()
            loss_module.wandb_features[f'additional_loss({loss_module.wandb_name})'].clear()

        if hasattr(runner.model.module.rpn_head.loss_bbox, 'wandb_features'):
            # save the loss and jsd loss of rpn_cls. rpn_cls consists of 5 feature maps.
            for wandb_feature, value in runner.model.module.rpn_head.loss_bbox.wandb_features.items():
                if isinstance(value, list):
                    for i, v in enumerate(value):
                        self.wandb.log({split + wandb_feature + '_layer' + str(i): v})
                else:
                    self.wandb.log({split + wandb_feature: value})
            loss_module = runner.model.module.rpn_head.loss_bbox
            if f'L1_loss({loss_module.wandb_name})' in loss_module.wandb_features:
                if isinstance(loss_module.wandb_features[f'L1_loss({loss_module.wandb_name})'], list):
                    loss_module.wandb_features[f'L1_loss({loss_module.wandb_name})'].clear()
            loss_module.wandb_features[f'additional_loss({loss_module.wandb_name})'].clear()

        if hasattr(runner.model.module.roi_head.bbox_head.loss_cls, 'wandb_features'):

            for wandb_feature, value in runner.model.module.roi_head.bbox_head.loss_cls.wandb_features.items():
                # print('wandb_feature: ', wandb_feature)
                if 'confusion_matrix' in wandb_feature:

                    value = value / (runner.model.module.roi_head.bbox_head.loss_cls.wandb_features[
                                         'matrix_sample_number(roi_cls)'] + 1e-8)
                    value = value.detach().cpu().numpy()
                    # save
                    plt = plot_matrix(value, normalize=True, title=wandb_feature)
                    plt.savefig(f"/ws/data/log/cityscapes/debug/{wandb_feature}.jpg")


                    self.wandb.log({split + wandb_feature: plt})
                elif 'matrix_sample_number' in wandb_feature:

                    value = value.detach().cpu().numpy()
                    plt = plot_matrix(value, normalize=True, title=wandb_feature)
                    self.wandb.log({split + wandb_feature: plt})

                elif 'distance' in wandb_feature:


                    value = value / (runner.model.module.roi_head.bbox_head.loss_cls.wandb_features[
                                         'iteration'] + 1e-8)
                    self.wandb.log({split + wandb_feature: value})
                else:
                    self.wandb.log({split + wandb_feature: value})

            loss_module = runner.model.module.roi_head.bbox_head.loss_cls
            if isinstance(loss_module.wandb_features[f'ce_loss({loss_module.wandb_name})'], list):
                loss_module.wandb_features[f'ce_loss({loss_module.wandb_name})'].clear()
            if isinstance(loss_module.wandb_features[f'additional_loss({loss_module.wandb_name})'], list):
                loss_module.wandb_features[f'additional_loss({loss_module.wandb_name})'].clear()

        if hasattr(runner.model.module.roi_head.bbox_head.loss_bbox, 'wandb_features'):
            for wandb_feature, value in runner.model.module.roi_head.bbox_head.loss_bbox.wandb_features.items():
                self.wandb.log({split + wandb_feature: value})
            loss_module = runner.model.module.roi_head.bbox_head.loss_bbox
            if f'smoothL1_loss({loss_module.wandb_name})' in loss_module.wandb_features:
                if isinstance(loss_module.wandb_features[f'smoothL1_loss({loss_module.wandb_name})'], list):
                    loss_module.wandb_features[f'smoothL1_loss({loss_module.wandb_name})'].clear()
            if f'L1_loss({loss_module.wandb_name})' in loss_module.wandb_features:
                if isinstance(loss_module.wandb_features[f'L1_loss({loss_module.wandb_name})'], list):
                    loss_module.wandb_features[f'L1_loss({loss_module.wandb_name})'].clear()
            if isinstance(loss_module.wandb_features[f'additional_loss({loss_module.wandb_name})'], list):
                loss_module.wandb_features[f'additional_loss({loss_module.wandb_name})'].clear()


    @master_only
    def after_run(self, runner):
        self.wandb.finish()

    def _check_priority(self, hook):
        """Check the if the priority of the hook is more than WandbLogger.

        Note that, a smaller priority will have bigger integer value and vice
        versa.
        """
        if isinstance(hook, CheckpointHook):
            if self.priority < hook.priority:
                self.log_checkpoint = False
                warnings.warn(
                    'The priority of CheckpointHook should '
                    'be more than WandbLogger to use log_checkpoint.',
                    UserWarning)
        elif isinstance(hook, (EvalHook, DistEvalHook)):
            if self.priority < hook.priority:
                self.num_eval_images = 0
                self.log_checkpoint_metadata = False
                self.log_eval_metrics = False
                warnings.warn(
                    'The priority of EvalHook should be more than '
                    'WandbLogger to log num_eval_images', UserWarning)
        else:
            print(f'The {hook.__name__} doesn\'t belong to CheckpointHook or '
                  'EvalHook.')

    def _log_ckpt_as_artifact(self,
                              path_to_model,
                              epoch,
                              aliases,
                              metadata=None):
        """Log model checkpoint as  W&B Artifact.

        Args:
            path_to_model (str): Path where model checkpoints are saved.
            epoch (int): The current epoch.
            aliases (list): List of the aliases associated with this artifact.
            metadata (dict, optional): Metadata associated with this artifact.
        """
        model_artifact = self.wandb.Artifact(
            f'run_{self.wandb.run.name}_model', type='model', metadata=metadata)
        model_artifact.add_file(f'{path_to_model}/epoch_{epoch+1}.pth')
        self.wandb.log_artifact(model_artifact, aliases=aliases)

    def _get_ckpt_metadata(self, runner):
        """"Get model checkpoint metadata."""
        if self.ckpt_hook.interval == self.eval_hook.interval:
            # results = self.eval_hook.results
            results = self.eval_hook.latest_results
        else:
            runner.logger.info(
                f'Evaluating for model checkpoint at epoch '
                f'{runner.epoch+1} which will be saved as W&B Artifact.')
            if isinstance(self.eval_hook, EvalHook):
                from mmdet.apis import single_gpu_test
                results = single_gpu_test(
                    runner.model, self.val_dataloader, show=False)
            elif isinstance(self.eval_hook, DistEvalHook):
                from mmdet.apis import multi_gpu_test
                results = multi_gpu_test(
                    runner.model, self.val_dataloader, gpu_collect=True)

        eval_results = self.val_dataset.evaluate(results, logger='silent')
        metadata = dict(epoch=runner.epoch + 1, **eval_results)
        return metadata

    def _is_best_ckpt(self, metadata):
        """Check if the current checkpoint has the best metric score.

        Args:
            metadata (dict): Metadata associated with the checkpoint.

        Returns:
            bool: Returns True, if the checkpoint has the best metric score.
        """
        keys = list(metadata.keys())
        map_metrics = [key for key in keys if 'mAP' in key]
        ar_metrics = [key for key in keys if 'AR' in key]
        if len(map_metrics) > 0:
            map_score = metadata.get(map_metrics[0], None)
            return self._is_best_score(map_score)
        elif len(ar_metrics) > 0:
            ar_score = metadata.get(ar_metrics[0], None)
            return self._is_best_score(ar_score)
        else:
            return False

    def _is_best_score(self, score):
        if score is None:
            return

        if score > self.best_score:
            self.best_score = score
            return True
        else:
            return False

    def _init_data_table(self):
        """Initialize the W&B Tables for validation data."""
        columns = ['image_name', 'image']
        self.data_table = self.wandb.Table(columns=columns)

    def _init_pred_table(self):
        """Initialize the W&B Tables for model evaluation."""
        columns = ['epoch', 'image_name', 'ground_truth', 'prediction'] + list(
            self.class_id_to_label.values())
        self.eval_table = self.wandb.Table(columns=columns)

    def _add_ground_truth(self):
        num_total_images = len(self.val_dataset)
        if self.num_eval_images > num_total_images:
            warnings.warn(
                'The num_eval_images is greater than the total number '
                'of validation samples. The complete validation set '
                'will be logged.', UserWarning)
        self.num_eval_images = min(self.num_eval_images, num_total_images)

        classes = self.val_dataset.get_classes()
        self.class_id_to_label = {id: name for id, name in enumerate(classes)}
        self.class_set = self.wandb.Classes([{
            'id': id,
            'name': name
        } for id, name in self.class_id_to_label.items()])

        from mmdet.datasets.pipelines import LoadImageFromFile
        img_loader = LoadImageFromFile()
        img_prefix = self.val_dataset.img_prefix

        for idx in range(self.num_eval_images):
            img_info = self.val_dataset.data_infos[idx]
            img_meta = img_loader(
                dict(img_info=img_info, img_prefix=img_prefix))

            # Get image and convert from BGR to RGB
            image = img_meta['img'][..., ::-1]
            image_name = img_info['filename']

            data_ann = self.val_dataset.get_ann_info(idx)
            bboxes = data_ann['bboxes']
            labels = data_ann['labels']

            box_data = []
            assert len(bboxes) == len(labels)
            for bbox, label in zip(bboxes, labels):
                box_data.append(
                    self._get_wandb_bbox(bbox, label, classes[label]))

            boxes = {
                'ground_truth': {
                    'box_data': box_data,
                    'class_labels': self.class_id_to_label
                }
            }

            self.data_table.add_data(
                image_name,
                self.wandb.Image(image, boxes=boxes, classes=self.class_set))

    def _log_predictions(self, results, epoch):
        table_idxs = self.data_table_ref.get_index()
        assert len(table_idxs) == self.num_eval_images

        for ndx in table_idxs:
            result = results[ndx]
            assert len(result) == len(self.class_id_to_label)

            box_data = []
            class_scores = []
            for label_id, bbox_scores in enumerate(result):
                if len(bbox_scores) != 0:
                    class_score = 0
                    count = 0
                    for bbox_score in bbox_scores:
                        confidence = float(bbox_score[4])
                        if confidence > 0.3:
                            class_score += confidence
                            count += 1
                            class_name = self.class_id_to_label[label_id]
                            box_data.append(
                                self._get_wandb_bbox(
                                    bbox_score, label_id,
                                    f'{class_name} {confidence:.2f}'))

                    class_scores.append(class_score / (count + 1e-6))
                else:
                    class_scores.append(0)

            boxes = {
                'predictions': {
                    'box_data': box_data,
                    'class_labels': self.class_id_to_label
                }
            }

            self.eval_table.add_data(
                epoch, self.data_table_ref.data[ndx][0],
                self.data_table_ref.data[ndx][1],
                self.wandb.Image(
                    self.data_table_ref.data[ndx][1],
                    boxes=boxes,
                    classes=self.class_set), *tuple(class_scores))

    def _get_wandb_bbox(self, bbox, label, box_caption):
        """Get structured dict for logging bounding boxes to W&B.

        Args:
            bbox (list): Bounding box coordinates in
                        (minX, minY, maxX, maxY) format.
            label (int): label id for that bounding box.
            box_caption (str): Caption for that bounding box.
            scale_factor (list): List rescaling factor for bounding box values.

        Returns:
            dict: Structured dict required for logging
                  that bounding box to W&B.
        """
        position = dict(
            minX=int(bbox[0]),
            minY=int(bbox[1]),
            maxX=int(bbox[2]),
            maxY=int(bbox[3]))

        if not isinstance(label, int):
            label = int(label)

        if not isinstance(box_caption, str):
            box_caption = str(box_caption)

        return {
            'position': position,
            'class_id': label,
            'box_caption': box_caption,
            'domain': 'pixel'
        }

    def _log_data_table(self):
        """Log the W&B Tables for validation data as artifact and calls
        `use_artifact` on it so that the evaluation table can use the reference
        of already uploaded images.

        This allows the data to be uploaded just once.
        """
        data_artifact = self.wandb.Artifact('val', type='dataset')
        data_artifact.add(self.data_table, 'val_data')

        self.wandb.run.use_artifact(data_artifact)
        data_artifact.wait()

        self.data_table_ref = data_artifact.get('val_data')

    def _log_eval_table(self):
        """Log the W&B Tables for model evaluation.

        The table will be logged multiple times creating new version. Use this
        to compare models at different intervals interactively.
        """
        pred_artifact = self.wandb.Artifact(
            f'run_{self.wandb.run.name}_pred', type='evaluation')
        pred_artifact.add(self.eval_table, 'eval_data')
        self.wandb.run.log_artifact(pred_artifact)
