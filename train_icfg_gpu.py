from utils.logger import setup_logger
from datasets import make_dataloader_vl_icfg_level_data as make_dataloader
from model.make_model_vl import make_model
from solver import make_optimizer_triplet as make_optimizer
# from solver import make_optimizer_uvt as make_optimizer
from solver.scheduler_factory import create_scheduler
# from loss import make_loss_triplet as make_loss
from loss import make_loss_entropy as make_loss
from processor import do_train_icfg_uvt_level_data as do_train
from processor import do_inference_cuhk_bert as do_inference
import random
import torch
import numpy as np
import os
import argparse
import subprocess
# from timm.scheduler import create_scheduler
from config import cfg
from model.backbones.tokenization_bert import BertTokenizer
from loss.cmps_loss import Loss
import torch.distributed as dist


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def resume_checkpoint(cfg, model, optimizer):
    path_c = os.path.join(cfg.OUTPUT_DIR, 'checkpoint.pth')
    if not os.path.exists(path_c):
        return model, 0, optimizer

    net_dict = torch.load(path_c)
    param_dict = net_dict['state_dict']
    for i in param_dict:
        model.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
    # model.load_state_dict(net_dict['state_dict'])
    start_epoch = net_dict['epoch'] + 1
    optimizer.load_state_dict(net_dict['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    return model, start_epoch, optimizer


def load_pre_checkpoint(root, model):
    path_c = os.path.join(root, 'checkpoint_best.pth')
    if not os.path.exists(path_c):
        return model

    net_dict = torch.load(path_c, map_location="cpu")
    start_epoch = net_dict['epoch']
    param_dict = net_dict['state_dict']
    for i in param_dict:
        try:
            model.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        except:
            continue
    # model.load_state_dict(net_dict['state_dict'])
    return model, start_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file", default="configs/ICFG/vit_base_uvt_bert.yml", help="path to config file", type=str)
    parser.add_argument("--gpu_id", default="0", type=str)
    parser.add_argument("--dataset", default="icfg_pedes", type=str)
    parser.add_argument("--pretrain_choice", default="imagenet", type=str)
    parser.add_argument("--loss_type", default="softmax_triplet", type=str, help='softmax, softmax_triplet')
    parser.add_argument("--model_name", default="transformer_uvt_img_txt_mask", type=str)
    parser.add_argument("--transformer_type", default="vit_base_patch16_224_uvt_img_txt_mask", type=str)
    parser.add_argument("--batch_size", default=28, type=int, help='28')
    parser.add_argument("--stride_size", default=12, type=int, help='12, 16')
    parser.add_argument("--data_dir", default="/data/reid/ICFG_PEDES/ICFG_PEDES", type=str)
    parser.add_argument("--logs_dir", default="/logs/20230521_icfg_gpu", type=str)

    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--test_mode", default=False, type=bool)
    parser.add_argument("--resume", default=False, type=bool)
    parser.add_argument("--use_pre", default=True, type=bool)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.MODEL.DEVICE_ID = args.gpu_id
    cfg.MODEL.PRETRAIN_CHOICE = args.pretrain_choice
    cfg.MODEL.TRANSFORMER_TYPE = args.transformer_type
    cfg.MODEL.NAME = args.model_name
    cfg.MODEL.STRIDE_SIZE = args.stride_size
    cfg.DATALOADER.SAMPLER = args.loss_type
    cfg.OUTPUT_DIR = args.logs_dir
    cfg.DATASETS.NAMES = args.dataset
    cfg.DATASETS.ROOT_DIR = args.data_dir
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size


    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    # if is_main_process() and output_dir and not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, test_loader, num_classes = make_dataloader(cfg)

    # tokenizer = BertTokenizer.from_pretrained(cfg.TEXT.TEXT_ENCODER)
    tokenizer = BertTokenizer.from_pretrained('./bert_tool/uncased_L-12_H-768_A-12')

    model = make_model(cfg, num_class=num_classes)

    if args.test_mode:
        if is_main_process():
            param_dict = torch.load(os.path.join(cfg.OUTPUT_DIR, 'checkpoint.pth'))['state_dict']
            for i in param_dict:
                model.state_dict()[i.replace('module.', '')].copy_(param_dict[i])

            print('Test:')
            R1, R5, R10, mAP = do_inference(cfg, model, test_loader, logger, tokenizer)
    else:
        loss_func = make_loss(cfg, num_classes=num_classes)
        loss_cmpm = Loss(num_classes=num_classes, feature_size=768, resume=False, epsilon=1e-8).cuda()

        optimizer = make_optimizer(cfg, model)
        scheduler = create_scheduler(cfg, optimizer)
        start_epoch = 0
        if args.use_pre:
            path_c = "./pretrain"
            model, pre_epoch = load_pre_checkpoint(path_c, model)
            print('resume checkpoint successful. pre_epoch: ', pre_epoch)
        if args.resume:
            model, start_epoch, optimizer = resume_checkpoint(cfg, model, optimizer)
            print('resume checkpoint successful. start_epoch: ', start_epoch)

        do_train(cfg, model, train_loader, test_loader, optimizer, scheduler, loss_func, loss_cmpm, tokenizer, args.local_rank, start_epoch, logger)







