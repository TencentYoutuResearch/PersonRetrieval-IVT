import torch
import numpy as np
import os
from utils.reranking import re_ranking
# from sklearn_cluster import *
import utils_knn.metrics as metrics


def _read_meta(fn):
    labels = list()
    lb_set = set()
    with open(fn) as f:
        for lb in f.readlines():
            lb = int(lb.strip())
            labels.append(lb)
            lb_set.add(lb)
    return np.array(labels), lb_set


def evaluate(gt_labels, pred_labels, metric='pairwise'):
    if isinstance(gt_labels, str) and isinstance(pred_labels, str):
        print('[gt_labels] {}'.format(gt_labels))
        print('[pred_labels] {}'.format(pred_labels))
        gt_labels, gt_lb_set = _read_meta(gt_labels)
        pred_labels, pred_lb_set = _read_meta(pred_labels)

        print('#inst: gt({}) vs pred({})'.format(len(gt_labels), len(pred_labels)))
        print('#cls: gt({}) vs pred({})'.format(len(gt_lb_set), len(pred_lb_set)))

    metric_func = metrics.__dict__[metric]
    result = metric_func(gt_labels, pred_labels)       # 0.8848, 0.8073, 0.84430

    return result

    # if isinstance(result, float):
    #     # print('{}: {:.4f}'.format(metric, result))
    #     return result
    # else:
    #     ave_pre, ave_rec, fscore = result       # 0.8848, 0.8073, 0.84430
    #     # print('{}ave_pre: {:.4f}, ave_rec: {:.4f}, fscore: {:.4f}'.format(metric, ave_pre, ave_rec, fscore))
    #     return result[-1]


def evaluation(pred_labels, labels, metrics):
    gt_labels_all = labels              # [10000,]
    pred_labels_all = pred_labels       # [10000,]
    metric_list = metrics.split(',')

    result_dict = {}
    for metric in metric_list:
        result = evaluate(gt_labels_all, pred_labels_all, metric)
        result_dict[metric] = result
    return result_dict



def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # dist_mat.addmm_(1, -2, qf, gf.t())
    dist_mat = dist_mat - 2 * torch.matmul(qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP



def eval_func_no_cam(distmat, q_pids, g_pids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP



class R1_mAP_CUHK():
    def __init__(self, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_CUHK, self).__init__()
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.feats_txt = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, feat_txt, pid = output
        self.feats.append(feat.cpu())
        self.feats_txt.append(feat_txt.cpu())
        self.pids.extend(np.asarray(pid))

    def compute(self):  # called after each epoch
        feats_img = torch.cat(self.feats, dim=0)        # [6148, 2048]
        feats_txt = torch.cat(self.feats_txt, dim=0)    # [6148, 2048]

        print("The test feature is normalized")
        feats_img = torch.nn.functional.normalize(feats_img, dim=1, p=2)     # [6148, 2048]
        feats_img = feats_img[::2]                      # [3074, 2048]
        feats_txt = torch.nn.functional.normalize(feats_txt, dim=1, p=2)     # [6148, 2048]

        self.pids_txt = np.asarray(self.pids)           # [6148,]
        self.pids_img = np.asarray(self.pids)[::2]      # [3074,]

        # cmc, mAP = self.compute_metric(feats_txt, self.pids_txt, feats_img, self.pids_img)
        cmc, mAP = self.compute_metric_sim(feats_txt, self.pids_txt, feats_img, self.pids_img)

        return cmc, mAP

    def compute_metric(self, qf, q_pids, gf, g_pids):  # called after each epoch
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)            # [6148, 3074]
        cmc, mAP = eval_func_no_cam(distmat, q_pids, g_pids)

        return cmc, mAP

    def compute_metric_sim(self, query_feature, query_label, gallery_feature, gallery_label):
        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        for ii in range(len(query_label)):
            ap_tmp, CMC_tmp = self.evaluate_ii(query_feature[ii], query_label[ii], gallery_feature, gallery_label)

            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
        CMC = CMC.float().data.cpu().numpy()
        CMC = CMC / len(query_label)
        mAP = ap / len(query_label)
        return CMC, mAP

    def evaluate_ii(self, qf, ql, gf, gl):      # [768,], 0, [3074, 768], [3074,]
        query = qf.view(-1, 1)                  # [768, 1]
        score = torch.mm(gf, query)             # [3074, 1]
        score = score.squeeze(1).cpu()          # [3074,]
        score = score.numpy()                   # [3074,]
        index = np.argsort(score)               # [3074,]
        index = index[::-1]                     # [3074,]
        query_index = np.argwhere(gl == ql)
        ap, cmc = self.compute_mAP(index, query_index)
        return ap, cmc

    def compute_mAP(self, index, good_index):
        ap = 0
        cmc = torch.IntTensor(len(index)).zero_()
        if good_index.size == 0:  # if empty
            cmc[0] = -1
            return ap, cmc
        # find good_index index
        ngood = len(good_index)
        mask = np.in1d(index, good_index)
        rows_good = np.argwhere(mask == True)
        rows_good = rows_good.flatten()

        cmc[rows_good[0]:] = 1
        for i in range(ngood):
            d_recall = 1.0 / ngood
            precision = (i + 1) * 1.0 / (rows_good[i] + 1)
            if rows_good[i] != 0:
                old_precision = i * 1.0 / rows_good[i]
            else:
                old_precision = 1.0
            ap = ap + d_recall * (old_precision + precision) / 2

        return ap, cmc


class R1_mAP_ICFG():
    def __init__(self, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_ICFG, self).__init__()
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.feats_txt = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, feat_txt, pid = output
        self.feats.append(feat.cpu())
        self.feats_txt.append(feat_txt.cpu())
        self.pids.extend(np.asarray(pid))

    def compute(self):  # called after each epoch
        feats_img = torch.cat(self.feats, dim=0)        # [19848, 2048]
        feats_txt = torch.cat(self.feats_txt, dim=0)    # [19848, 2048]

        print("The test feature is normalized")
        feats_img = torch.nn.functional.normalize(feats_img, dim=1, p=2)     # [19848, 2048]
        feats_txt = torch.nn.functional.normalize(feats_txt, dim=1, p=2)     # [19848, 2048]

        self.pids_txt = np.asarray(self.pids)           # [19848,]
        self.pids_img = np.asarray(self.pids)           # [19848,]

        # cmc, mAP = self.compute_metric(feats_txt, self.pids_txt, feats_img, self.pids_img)
        cmc, mAP = self.compute_metric_sim(feats_txt, self.pids_txt, feats_img, self.pids_img)

        return cmc, mAP

    def compute_metric(self, qf, q_pids, gf, g_pids):  # called after each epoch
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)            # [6148, 3074]
        cmc, mAP = eval_func_no_cam(distmat, q_pids, g_pids)

        return cmc, mAP

    def compute_metric_sim(self, query_feature, query_label, gallery_feature, gallery_label):
        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        for ii in range(len(query_label)):
            ap_tmp, CMC_tmp = self.evaluate_ii(query_feature[ii], query_label[ii], gallery_feature, gallery_label)

            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
        CMC = CMC.float().data.cpu().numpy()
        CMC = CMC / len(query_label)
        mAP = ap / len(query_label)
        return CMC, mAP

    def evaluate_ii(self, qf, ql, gf, gl):      # [768,], 0, [3074, 768], [3074,]
        query = qf.view(-1, 1)                  # [768, 1]
        score = torch.mm(gf, query)             # [3074, 1]
        score = score.squeeze(1).cpu()          # [3074,]
        score = score.numpy()                   # [3074,]
        index = np.argsort(score)               # [3074,]
        index = index[::-1]                     # [3074,]
        query_index = np.argwhere(gl == ql)
        ap, cmc = self.compute_mAP(index, query_index)
        return ap, cmc

    def compute_mAP(self, index, good_index):
        ap = 0
        cmc = torch.IntTensor(len(index)).zero_()
        if good_index.size == 0:  # if empty
            cmc[0] = -1
            return ap, cmc
        # find good_index index
        ngood = len(good_index)
        mask = np.in1d(index, good_index)
        rows_good = np.argwhere(mask == True)
        rows_good = rows_good.flatten()

        cmc[rows_good[0]:] = 1
        for i in range(ngood):
            d_recall = 1.0 / ngood
            precision = (i + 1) * 1.0 / (rows_good[i] + 1)
            if rows_good[i] != 0:
                old_precision = i * 1.0 / rows_good[i]
            else:
                old_precision = 1.0
            ap = ap + d_recall * (old_precision + precision) / 2

        return ap, cmc
















