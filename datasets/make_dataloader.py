import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
from utils.randaugment import RandomAugment
from .bases import ImageDataset, ImageDatasetUVT, ImageDatasetUVTLevel, ImageDataset_IMGTXT_PRE, ImageDatasetLevelData, ImageDatasetLevelICFG, ImageDatasetPRE
from .bases import ImageDatasetLaST, ImageDatasetLevelRSTP, ImageDatasetPath

from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler, RandomIdentitySamplerCUHK, RandomIdentitySamplerPRE
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP, RandomIdentitySampler_DDP_CUHK, RandomIdentitySampler_DDP_PRE
import torch.distributed as dist
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi
from .last import LaST
from .cuhk_pedes import CUHK_PEDES
from .pedes import CuhkPedes
from .rstp_reid import RSTPReID
from .icfg_pedes import ICFG_PEDES
from .pretrain_cuhk import PRETRAIN_CUHK
from .cuhk_pedes_level import CUHK_PEDES_LEVEL

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
    'last': LaST,
    'cuhk_pedes': CUHK_PEDES,
    'pedes': CuhkPedes,
    'cuhk_pedes_level': CUHK_PEDES_LEVEL,
    'rstp_reid': RSTPReID,
    'icfg_pedes': ICFG_PEDES,
    'pretrain_cuhk': PRETRAIN_CUHK,

}




def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths



def sample_collate_fn_level_rstp(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs1, imgs2, imgs3, caption, pids, img_paths, word = zip(*batch)
    pids = torch.from_numpy(np.array(pids))
    imgs1 = torch.stack(imgs1, dim=0)
    imgs2 = torch.stack(imgs2, dim=0)
    imgs3 = torch.stack(imgs3, dim=0)
    return imgs1, imgs2, imgs3, caption, pids, img_paths, word


def sample_collate_fn_uvt(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, caption, pids, img_paths, token = zip(*batch)
    pids = torch.from_numpy(np.array(pids))
    return torch.stack(imgs, dim=0), caption, pids, img_paths, token

def sample_collate_fn_level_data(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs1, imgs2, imgs3, caption, pids, img_paths, token = zip(*batch)
    pids = torch.from_numpy(np.array(pids))
    imgs1 = torch.stack(imgs1, dim=0)
    imgs2 = torch.stack(imgs2, dim=0)
    imgs3 = torch.stack(imgs3, dim=0)
    return imgs1, imgs2, imgs3, caption, pids, img_paths, token



def sample_collate_fn_img_txt_pre(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, caption, pids, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, img_paths = zip(*batch)
    pids = torch.from_numpy(np.array(pids))
    text_ids = torch.tensor(text_ids, dtype=torch.long)
    text_ids_masked = torch.tensor(text_ids_masked, dtype=torch.long)
    text_atts = torch.tensor(text_atts, dtype=torch.long)
    masked_pos = torch.tensor(masked_pos, dtype=torch.long)
    masked_ids = torch.tensor(masked_ids, dtype=torch.long)
    return torch.stack(imgs, dim=0), caption, pids,text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, img_paths



def sample_collate_fn_pre(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, caption, pids, img_paths = zip(*batch)
    pids = torch.from_numpy(np.array(pids))
    return torch.stack(imgs, dim=0), caption, pids, img_paths


def sample_collate_fn_path(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, cams, img_paths = zip(*batch)
    pids = torch.from_numpy(np.array(pids))
    cams = torch.from_numpy(np.array(cams))
    return torch.stack(imgs, dim=0), pids, cams, img_paths










def make_dataloader_vl_cuhk_level_data(cfg, file='cuhk_pedes_att.json'):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, file=file)

    train_set = ImageDatasetLevelData(dataset.root, dataset.train, train_transforms)
    num_classes = dataset.num_train_pids        # 11003

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            # mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()

            data_sampler = RandomIdentitySampler_DDP_CUHK(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, cfg.SOLVER.IMS_PER_BATCH, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=sample_collate_fn_level_data,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySamplerCUHK(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=sample_collate_fn_level_data
            )

    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            # mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()

            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, num_workers=num_workers,
                collate_fn=sample_collate_fn_level_data, sampler=train_sampler)

        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                collate_fn=sample_collate_fn_level_data
            )

    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    valset = ImageDatasetUVTLevel(dataset.root, dataset.val, val_transforms)
    val_loader = DataLoader(
        valset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=sample_collate_fn_uvt
    )

    testset = ImageDatasetUVTLevel(dataset.root, dataset.test, val_transforms)
    test_loader = DataLoader(
        testset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=sample_collate_fn_uvt
    )

    return train_loader, val_loader, test_loader, num_classes



def make_dataloader_vl_icfg_level_data(cfg, file='ICFG_PEDES_VISTXT.json'):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, file=file)

    train_set = ImageDatasetLevelICFG(dataset.root, dataset.train, train_transforms)
    num_classes = dataset.num_train_pids        # 11003

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            # mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()

            data_sampler = RandomIdentitySampler_DDP_CUHK(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, cfg.SOLVER.IMS_PER_BATCH, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=sample_collate_fn_level_rstp,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySamplerCUHK(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=sample_collate_fn_level_rstp
            )

    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            # mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()

            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, num_workers=num_workers,
                collate_fn=sample_collate_fn_level_rstp, sampler=train_sampler)

        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                collate_fn=sample_collate_fn_level_rstp
            )

    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))


    testset = ImageDatasetUVT(dataset.root, dataset.test, val_transforms)
    test_loader = DataLoader(
        testset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=sample_collate_fn_uvt
    )

    return train_loader, test_loader, num_classes



def make_dataloader_vl_rstp_level_data(cfg, file='rstpreid_vistxt.json'):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, file=file)

    train_set = ImageDatasetLevelRSTP(dataset.root, dataset.train, train_transforms)
    num_classes = dataset.num_train_pids        # 11003

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            # mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()

            data_sampler = RandomIdentitySampler_DDP_CUHK(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, cfg.SOLVER.IMS_PER_BATCH, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=sample_collate_fn_level_rstp,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySamplerCUHK(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=sample_collate_fn_level_rstp
            )

    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            # mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()

            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, num_workers=num_workers,
                collate_fn=sample_collate_fn_level_rstp, sampler=train_sampler)

        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                collate_fn=sample_collate_fn_level_rstp
            )

    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    valset = ImageDatasetUVT(dataset.root, dataset.val, val_transforms)
    val_loader = DataLoader(
        valset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=sample_collate_fn_uvt
    )

    testset = ImageDatasetUVT(dataset.root, dataset.test, val_transforms)
    test_loader = DataLoader(
        testset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=sample_collate_fn_uvt
    )

    return train_loader, val_loader, test_loader, num_classes




def make_dataloader_vl_img_txt_pre(cfg, file='cuhk_pedes_vistxt.json'):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, file=file)

    train_set = ImageDataset_IMGTXT_PRE(dataset.root, dataset.train, train_transforms)

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            # mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()

            data_sampler = RandomIdentitySampler_DDP_PRE(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, cfg.SOLVER.IMS_PER_BATCH, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=sample_collate_fn_img_txt_pre,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySamplerCUHK(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=sample_collate_fn_img_txt_pre
            )

    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            # mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()

            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, num_workers=num_workers,
                collate_fn=sample_collate_fn_img_txt_pre, sampler=train_sampler)

            # data_sampler = RandomIdentitySampler_DDP_PRE(dataset.train, cfg.SOLVER.IMS_PER_BATCH, 1)
            # batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            # train_loader = torch.utils.data.DataLoader(
            #     train_set,
            #     num_workers=num_workers,
            #     batch_sampler=batch_sampler,
            #     collate_fn=sample_collate_fn_img_txt_pre,
            #     pin_memory=True,
            # )

        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                collate_fn=sample_collate_fn_img_txt_pre
            )

    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    testset = ImageDatasetPRE(dataset.root, dataset.test, val_transforms)
    test_loader = DataLoader(
        testset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=sample_collate_fn_pre
    )

    return train_loader, test_loader, 1000



def make_dataloader_last(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    test_set = ImageDataset(dataset.query_test + dataset.gallery_test, val_transforms)
    test_loader = DataLoader(
        test_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, test_loader, len(dataset.query), len(dataset.query_test), num_classes, cam_num, view_num




def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num



def make_dataloader_market(cfg):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set = ImageDatasetPath(dataset.dataset_dir, dataset.train, train_transforms)
    num_classes = dataset.num_train_pids  # 751

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()

            data_sampler = RandomIdentitySampler_DDP_CUHK(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=sample_collate_fn_path,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySamplerCUHK(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=sample_collate_fn_path
            )

    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=sample_collate_fn_path
        )

    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    testset = ImageDatasetPath(dataset.dataset_dir, dataset.query + dataset.gallery, val_transforms)
    test_loader = DataLoader(
        testset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=sample_collate_fn_path
    )

    return train_loader, test_loader, len(dataset.query), num_classes























