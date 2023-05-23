import torchvision.transforms as T
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import collections
import sys
from torch.utils.data import DataLoader
from datasets import init_dataset
from torch.utils.data.sampler import Sampler
import copy
from collections import defaultdict
import numpy as np
import torch
import random
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, ToPILImage
import cv2
import os
from ignite.metrics import Metric

def train_collate_fn(batch):
    imgs, pids, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    imgs, pids, camids = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids

def train_collate_fn_path(batch):
    imgs, pids, _, pathes = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, pathes


def val_collate_fn_path(batch):
    imgs, pids, camids, paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, paths

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img



def read_image_s(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)       # 12185

    def __getitem__(self, index):
        try:
            img_path, pid, camid = self.dataset[index]
            img = read_image(img_path)

            if self.transform is not None:
                img = self.transform(img)       # [3, 256, 128]
        except:
            print(index)

        return img, pid, camid






class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


def build_transforms(cfg, is_train=True, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225]):
    normalize_transform = T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)

    if is_train:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.RandomHorizontalFlip(),
            T.Pad(10),
            T.RandomCrop([cfg.height, cfg.width]),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.ToTensor(),
            normalize_transform,
        ])
    else:
        transform = T.Compose([
            T.Resize([cfg.height, cfg.width]),
            T.ToTensor(),
            normalize_transform
        ])

    return transform

def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = 8

    dataset = init_dataset(cfg.dataset, root=cfg.dir)

    num_classes = dataset.num_train_pids        # market:751,  msmst: 1041
    train_set = ImageDataset(dataset.train, train_transforms)

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size,
        sampler=RandomIdentitySampler(dataset.train, cfg.batch_size, cfg.img_per_id),           # 64, 4
        num_workers=num_workers, collate_fn=train_collate_fn
    )

    test_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size_test, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, test_loader, len(dataset.query), num_classes


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):     # [3368, 15913], [3368,], [15913,]
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)   # [3368, 15913]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)    # [3368, 15913]

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):       # 3368
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)      # [15913,]
        keep = np.invert(remove)      # [15913,]

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]       # [15908,]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()          # [15908,]
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()         # 14
        tmp_cmc = orig_cmc.cumsum()      # [15908,], [0,0,0,...,14,14,14]
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]     # [15908,]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc    # [15908,]
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q      # [50,]
    mAP = np.mean(all_AP)

    del indices, matches

    return all_cmc, mAP


class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reset()

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)       # [19281, 2048]
        if self.feat_norm == 'yes':
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]                           # [3368, 2048]
        q_pids = np.asarray(self.pids[:self.num_query])       # [3368,]
        q_camids = np.asarray(self.camids[:self.num_query])   # [3368,]
        # gallery
        gf = feats[self.num_query:]                           # [15913, 2048]
        g_pids = np.asarray(self.pids[self.num_query:])       # [15913,]
        g_camids = np.asarray(self.camids[self.num_query:])   # [15913,]
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())                     # [3368, 15913]
        distmat = distmat - 2 * torch.matmul(qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP

def fliplr(img):
    # flip horizontal
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.cpu().index_select(3, inv_idx)
    return img_flip.cuda()


def norm(f):
    # f = f.squeeze()
    fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
    f = f.div(fnorm.expand_as(f))
    return f


def inference_base(model, test_loader, num_query):
    print('Test')
    model.eval()
    metric = R1_mAP(num_query, 500)
    with torch.no_grad():
        for ii, batch in enumerate(test_loader):        # len(test_loader)=151
            data, pid, cmp, path = batch              # [128, 3, 256, 128]
            data = data.to("cuda") if torch.cuda.device_count() >= 1 else data
            f1 = model.forward_img(data)  # [128, 3840]
            f2 = model.forward_img(fliplr(data))  # [128, 3840]
            f = 0.5 * (f1 + f2)  # [128, 2048]
            f = norm(f)  # [128, 2048]
            metric.update([f, pid, cmp])
        cmc, mAP = metric.compute()
        return mAP, cmc[0], cmc[4], cmc[9], cmc[19]



