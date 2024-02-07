import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def analyze_representations_2input_sample(logits_clean, logits_aug1, lambda_weight=12, temper=1.0, reduction='batchmean'):
    '''
    logging representations by jsdv4 and L2 distance
    3 inputs
    '''

    device = logits_clean.device

    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]

    logits_clean = logits_clean.detach()
    logits_aug1 = logits_aug1.detach()

    # logging
    batch_size = logits_clean.size()[0]
    temper = 1.0

    # softmax
    p_clean, p_aug1  = F.softmax(logits_clean / temper, dim=1), \
                       F.softmax(logits_aug1 / temper, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1) / 2., 1e-7, 1).log()

    # JSD matrix
    jsd_matrix = (make_matrix(p_clean, p_mixture, criterion=nn.KLDivLoss(reduction='none'), reduction='sum') + \
                  make_matrix(p_aug1, p_mixture, criterion=nn.KLDivLoss(reduction='none'), reduction='sum')) / 2

    # MSE matrix
    mse_matrix = make_matrix(logits_clean, logits_aug1, criterion=nn.MSELoss(reduction='none'), reduction='mean')
    l2_matrix = torch.sqrt(mse_matrix)

    # Cosine Similarity matrix
    cs_matrix = make_matrix(logits_clean, logits_aug1, criterion=nn.CosineSimilarity(dim=1), reduction='none')
    cs_matrix = cs_matrix.squeeze(dim=-1)


    features = {
                'sample_matrix_jsd': jsd_matrix.detach().cpu().numpy(),
                'sample_matrix_l2': l2_matrix.detach().cpu().numpy(),
                'sample_matrix_cs': cs_matrix.detach().cpu().numpy(),
                }

    return features


def analyze_representations_2input(logits_clean, logits_aug1, labels=None, lambda_weight=12, temper=1.0, reduction='batchmean'):
    '''
    logging representations by jsdv4 and L2 distance
    3 inputs
    '''

    device = logits_clean.device
    targets = labels

    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]

    logits_clean = logits_clean.detach()
    logits_aug1 = logits_aug1.detach()

    # logging
    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    # mask
    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_same_triuu = mask_same_class * mask_triuu
    mask_diff_class = 1 - mask_same_class  # [B, B]
    mask_diff_triuu = mask_diff_class * mask_triuu

    # MSE matrix
    mse_matrix = make_matrix(logits_clean, logits_aug1, criterion=nn.MSELoss(reduction='none'), reduction='mean')

    mse_matrix_same_instance = mse_matrix * mask_same_instance
    mse_distance = mse_matrix_same_instance.sum()

    mse_matrix_diff_class = mse_matrix * mask_diff_triuu
    mse_distance_diff_class = mse_matrix_diff_class.sum()

    mse_matrix_same_class = mse_matrix * mask_same_triuu
    mse_distance_same_class = mse_matrix_same_class.sum()

    # Cosine Similarity matrix
    cs_matrix = make_matrix(logits_clean, logits_aug1, criterion=nn.CosineSimilarity(dim=1), reduction='none')
    cs_matrix = cs_matrix.squeeze(dim=-1)

    cs_matrix_same_instance = cs_matrix * mask_same_instance
    cs_distance = cs_matrix_same_instance.sum()

    cs_matrix_diff_class = cs_matrix * mask_diff_triuu
    cs_distance_diff_class = cs_matrix_diff_class.sum()

    cs_matrix_same_class = cs_matrix * mask_same_triuu
    cs_distance_same_class = cs_matrix_same_class.sum()

    # class-wise distance
    confusion_matrix_l2 = torch.zeros(9, 9)
    confusion_matrix_cs = torch.zeros(9, 9)
    confusion_matrix_sample_number = torch.zeros(9, 9)

    B, _ = targets.size()
    targets1 = targets.repeat(1, B).unsqueeze(0)
    targets2 = targets.T.repeat(B, 1).unsqueeze(0)
    target_matrix = torch.cat([targets1, targets2], dim=0) # class index of batch sampe (2, 512, 512) (target, target) tuple


    for i in range(9):
        for j in range(9):
            a = target_matrix[0, :, :] == i
            b = target_matrix[1, :, :] == j
            class_mask = a & b

            # class_jsd_matrix = jsd_matrix * class_mask
            class_mse_matrix = mse_matrix * class_mask
            class_cs_matrix = cs_matrix * class_mask

            # confusion_matrix_jsd[i, j] = class_jsd_matrix.sum()
            confusion_matrix_l2[i, j] = torch.sqrt(class_mse_matrix).sum()
            confusion_matrix_cs[i, j] = class_cs_matrix.sum()
            confusion_matrix_sample_number[i, j] = class_mask.sum()


    features = {
                'mse_distance': mse_distance.detach().cpu().numpy(),
                'mse_distance_diff_class': mse_distance_diff_class.detach().cpu().numpy(),
                'mse_distance_same_class': mse_distance_same_class.detach().cpu().numpy(),
                'confusion_matrix_l2': confusion_matrix_l2.detach().cpu().numpy(),
                'confusion_matrix_cs': confusion_matrix_cs.detach().cpu().numpy(),
                'matrix_sample_number': confusion_matrix_sample_number.detach().cpu().numpy(),
                }

    return features


def supcontrast_mask(logits_anchor, logits_contrast, targets, mask_anchor, mask_contrast, lambda_weight=0.1, temper=0.07):

    if logits_anchor.size(0) == 0:
        print("no foreground")
        loss = 0
    else:
        base_temper = temper

        logits_anchor, logits_contrast = F.normalize(logits_anchor, dim=1), F.normalize(logits_contrast, dim=1)

        anchor_dot_contrast = torch.div(torch.matmul(logits_anchor, logits_contrast.T), temper)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * mask_contrast
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask_anchor * log_prob).sum(1) / (mask_anchor.sum(1) + 1e-8)
        loss = - (temper / base_temper) * mean_log_prob_pos
        loss = loss.mean()

    return loss


def supcontrast(logits_clean, labels=None, num_views=2, lambda_weight=0.1, temper=0.07, min_samples=10):
    """
        supcontrast loss
        pull same fg class, push diff class fg, push fg vs bg, push diff instance bg
        create mask anchor and masks contrast
        input: only clean logit
        mask (mask anchor): augmented instance, original same class, augmented same class [3*B, 3*B]
        logits_mask (mask contrast): Self-instance case should be excluded
        mask_fg: only fg
        mask_anchor: same target
        mask_anchor_except_eye: same target except self-case
        mask_anchor_fg: same fg target except self-case
        mask_contrast_except_eye: all except self-case

    """

    device = logits_clean.device
    batch_size = logits_clean.size()[0]

    assert num_views == 2, "Only num_views 2 and batch_size 2 case are supported."
    ori_size = 512 * num_views   # 1024
    rp_total_size = batch_size % ori_size  # num_proposals
    rp_size = rp_total_size // num_views     # num proposals 10 case, num_view 2, batch_size 2 -> rp_size 20

    targets = labels
    targets = targets.contiguous().view(-1, 1)

    inds, _ = (targets != targets.max()).nonzero(as_tuple=True)

    fg = (targets != targets.max()).float()
    bg = (targets == targets.max()).float()
    mask_bg = torch.matmul(bg, bg.T)
    mask_same_instance = torch.zeros([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_eye_ori = torch.eye(ori_size, dtype=torch.float32).to(device)
    mask_eye_rp = torch.eye(rp_size, dtype=torch.float32).to(device)
    mask_same_instance[:ori_size, ori_size:ori_size*2] = mask_eye_ori
    mask_same_instance[ori_size:ori_size*2, :ori_size] = mask_eye_ori
    mask_same_instance[ori_size*2+rp_size:ori_size*2+rp_size*2, ori_size*2:ori_size*2+rp_size] = mask_eye_rp
    mask_same_instance[ori_size*2:ori_size*2+rp_size, ori_size*2+rp_size:ori_size*2+rp_size*2] = mask_eye_rp
    mask_anchor_bg = mask_same_instance * mask_bg

    if inds.size(0) > min_samples:

        mask_fg = torch.matmul(fg, fg.T)
        mask_eye = torch.eye(batch_size, dtype=torch.float32).to(device)
        mask_anchor = torch.eq(targets, targets.T).float()  # [B, B]
        mask_anchor_except_eye = mask_anchor - mask_eye

        mask_anchor_fg = mask_anchor_except_eye * mask_fg
        mask_anchor = mask_anchor_fg + mask_anchor_bg
        mask_anchor = mask_anchor.detach()

        assert ((mask_anchor!=0)&(mask_anchor!=1)).float().sum().item() == 0, "mask_anchor error"
        mask_contrast = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
        mask_contrast_except_eye = mask_contrast - mask_eye
        mask_contrast_except_eye = mask_contrast_except_eye.detach()

        loss = supcontrast_mask(logits_clean, logits_clean, targets,
                                mask_anchor, mask_contrast_except_eye, lambda_weight, temper)
    else:
        loss = torch.tensor(0., device=device, dtype=torch.float32)

    return loss

def supcontrast_yolo(logits_clean, labels=None, num_views=2, lambda_weight=0.1, temper=0.07, min_samples=10):
   """
       supcontrast loss
       pull same fg class, push diff class fg, push fg vs bg, push diff instance bg
       create mask anchor and masks contrast
       input: only clean logit
       mask (mask anchor): augmented instance, original same class, augmented same class [3*B, 3*B]
       logits_mask (mask contrast): Self-instance case should be excluded
       mask_fg: only fg
       mask_anchor: same target
       mask_anchor_except_eye: same target except self-case
       mask_anchor_fg: same fg target except self-case
       mask_contrast_except_eye: all except self-case

       contrastive loss for yolo model
       do not use oagrb model

   """

   device = logits_clean.device
   batch_size = logits_clean.size()[0]

   assert num_views == 2, "Only num_views 2 and batch_size 2 case are supported."
   ori_size = batch_size // 2   # 1024
   rp_total_size = batch_size % ori_size    # num_proposals
   rp_size = rp_total_size // num_views     # num proposals 10 case, num_view 2, batch_size 2 -> rp_size 20

   targets = labels
   targets = targets.contiguous().view(-1, 1)

   inds, _ = (targets != targets.max()).nonzero(as_tuple=True)

   fg = (targets != targets.max()).float()
   bg = (targets == targets.max()).float()
   mask_bg = torch.matmul(bg, bg.T)
   mask_same_instance = torch.zeros([batch_size, batch_size], dtype=torch.float32).to(device)
   mask_eye_ori = torch.eye(ori_size, dtype=torch.float32).to(device)
   mask_eye_rp = torch.eye(rp_size, dtype=torch.float32).to(device)
   mask_same_instance[:ori_size, ori_size:ori_size*2] = mask_eye_ori
   mask_same_instance[ori_size:ori_size*2, :ori_size] = mask_eye_ori
   mask_same_instance[ori_size*2+rp_size:ori_size*2+rp_size*2, ori_size*2:ori_size*2+rp_size] = mask_eye_rp
   mask_same_instance[ori_size*2:ori_size*2+rp_size, ori_size*2+rp_size:ori_size*2+rp_size*2] = mask_eye_rp
   mask_anchor_bg = mask_same_instance * mask_bg

   if inds.size(0) > min_samples:

       mask_fg = torch.matmul(fg, fg.T)
       mask_eye = torch.eye(batch_size, dtype=torch.float32).to(device)
       mask_anchor = torch.eq(targets, targets.T).float()  # [B, B]
       mask_anchor_except_eye = mask_anchor - mask_eye

       mask_anchor_fg = mask_anchor_except_eye * mask_fg
       mask_anchor = mask_anchor_fg + mask_anchor_bg
       mask_anchor = mask_anchor.detach()

       assert ((mask_anchor!=0)&(mask_anchor!=1)).float().sum().item() == 0, "mask_anchor error"
       mask_contrast = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
       mask_contrast_except_eye = mask_contrast - mask_eye
       mask_contrast_except_eye = mask_contrast_except_eye.detach()

       loss = supcontrast_mask(logits_clean, logits_clean, targets,
                               mask_anchor, mask_contrast_except_eye, lambda_weight, temper)
   else:
       loss = torch.tensor(0., device=device, dtype=torch.float32)

   return loss

