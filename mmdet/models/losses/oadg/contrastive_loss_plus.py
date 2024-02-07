# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...builder import LOSSES
from mmdet.models.losses.oadg.contrastive_loss import supcontrast


@LOSSES.register_module()
class ContrastiveLossPlus(nn.Module):

    def __init__(self,
                 loss_weight=1,
                 temperature=0.07,
                 num_views=2,
                 normalized_input=True,
                 min_samples=10,
                 **kwargs):
        """ContrastiveLossPlus."""
        super(ContrastiveLossPlus, self).__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.num_views = num_views
        self.normalized_input = normalized_input
        self.min_samples = min_samples
        self.kwargs = kwargs

        self.loss = supcontrast

    def forward(self, cont_feats, labels):
        """Forward function.

        Args:
        Returns:
            torch.Tensor: The calculated loss.
        """
        if len(cont_feats) == 0:
            return torch.zeros(1)
        if self.normalized_input:
            cont_feats = F.normalize(cont_feats, dim=1)

        # cont_feats: [2048+random_feats, 256], labels_: [2048+random_feats, ]
        if len(cont_feats) != len(labels): # random proposal case
            random_proposal_len = len(cont_feats) - len(labels)
            random_proposal_targets = labels[-1, :].repeat(random_proposal_len, 1)
            labels = torch.cat([labels, random_proposal_targets], dim=0)

        loss = self.loss(cont_feats, labels, num_views= self.num_views, temper=self.temperature, min_samples=self.min_samples)
        return self.loss_weight * loss

