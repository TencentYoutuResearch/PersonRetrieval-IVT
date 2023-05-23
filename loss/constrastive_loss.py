"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature              # 0.07
        self.contrast_mode = contrast_mode          # 'all'
        self.base_temperature = base_temperature    # 0.07

    def forward(self, features, labels=None, mask=None):     # [64, 2, 128], [64,]
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]      # 64
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)            # [64, 1]
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)        # [64, 64] <- 0, 1
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]          # 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)      # [128, 128]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature       # [128, 128]
            anchor_count = contrast_count           # 2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)       # [128, 128]
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)     # [128, 1]
        logits = anchor_dot_contrast - logits_max.detach()       # [128, 128]

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)         # [128, 128]
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)   # [128, 128], eye matrix
        mask = mask * logits_mask        # [128, 128]

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask            # [128, 128]
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))        # [128, 128]

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)      # [128,]

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos      # [128,]
        loss = loss.view(anchor_count, batch_size).mean()           # 4.9843

        return loss





class PairCosineLoss(nn.Module):
    def __init__(self,):
        super(PairCosineLoss, self).__init__()

    def forward(self, feat_vis, feat_txp, feat_txn):            # [64, 768], [64, 768]
        batch_size = feat_vis.shape[0]                          # 64

        feat_vis = F.normalize(feat_vis, dim=-1)                # [64, 768]
        feat_txp = F.normalize(feat_txp, dim=-1)                # [64, 768]
        feat_txn = F.normalize(feat_txn, dim=-1)                # [64, 768]

        sim_p = F.cosine_similarity(feat_vis, feat_txp).mean()
        sim_n = F.cosine_similarity(feat_vis, feat_txn).mean()

        loss = 2.0 - sim_p + sim_n

        return loss




class PairRankingLoss(nn.Module):
    def __init__(self,):
        super(PairRankingLoss, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def dist_vec(self, feat_vis, feat_txt):         # [16, 768], [16, 768]
        xx = torch.pow(feat_vis, 2).sum(1, keepdim=True)        # [16, 1]
        yy = torch.pow(feat_txt, 2).sum(1, keepdim=True)        # [16, 1]
        dist = xx + yy        # [16, 1]
        dist = dist - 2 * torch.mul(feat_vis, feat_txt).sum(1, keepdim=True)        # [16, 1]
        dist = dist.clamp(min=1e-12).sqrt()  # [16, 1], for numerical stability
        return dist

    def forward(self, feat_vis, feat_txp, feat_txn):  # [64, 768], [64, 768]
        batch_size = feat_vis.shape[0]  # 64

        feat_vis = F.normalize(feat_vis, dim=-1)        # [64, 768]
        feat_txp = F.normalize(feat_txp, dim=-1)        # [64, 768]
        feat_txn = F.normalize(feat_txn, dim=-1)        # [64, 768]

        dist_p = self.dist_vec(feat_vis, feat_txp)      # [64, 1]
        dist_n = self.dist_vec(feat_vis, feat_txn)      # [64, 1]

        y = dist_p.new().resize_as_(dist_p).fill_(1)    # [64, 1]
        loss = self.ranking_loss(dist_n - dist_p, y)    # 0.6949

        return loss




# class PairWiseLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super(PairWiseLoss, self).__init__()
#         self.temperature = temperature              # 0.07
#
#     def forward(self, feat_vis, feat_txp, feat_txn):  # [64, 768], [64, 768]
#         batch_size = feat_vis.shape[0]  # 64
#
#         feat_vis = F.normalize(feat_vis, dim=-1)            # [64, 768]
#         feat_txp = F.normalize(feat_txp, dim=-1)            # [64, 768]
#         feat_txn = F.normalize(feat_txn, dim=-1)            # [64, 768]
#
#         sim_p = torch.div(torch.mul(feat_vis, feat_txp).sum(dim=-1, keepdim=True), self.temperature)      # [64, 1]
#         sim_n = torch.div(torch.mul(feat_vis, feat_txn).sum(dim=-1, keepdim=True), self.temperature)      # [64, 1]
#
#         exp_logits = torch.exp(sim_p)                       # [64, 1]
#         log_prob_p = sim_p - torch.log(exp_logits)          # [64, 1]
#         loss_p = -log_prob_p.mean()
#
#         exp_logits_ = torch.exp(sim_n)                      # [64, 1]
#         log_prob_n = sim_n - torch.log(exp_logits_)         # [64, 1]
#         loss_n = log_prob_n.mean()
#
#         loss = loss_p + loss_n
#         return loss


if __name__ == '__main__':
    feat_vis = torch.randn((16, 768))
    feat_txp = torch.randn((16, 768))
    feat_txn = torch.randn((16, 768))

    loss_fun = PairCosineLoss()
    # loss_fun = PairRankingLoss()
    # loss_fun = PairWiseLoss()

    loss = loss_fun(feat_vis, feat_txp, feat_txn)
    print('loss = ', loss.item())


    print('finish')