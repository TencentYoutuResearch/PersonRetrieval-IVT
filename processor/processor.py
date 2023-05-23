import logging
import os
import time
import copy
import random
import torch
import torch.nn as nn
import errno
import shutil
from utils.meter import AverageMeter
from utils.metrics import *     # R1_mAP_eval, R1_mAP_eval_vl
from torch.cuda import amp
import torch.distributed as dist
from tensorboardX import SummaryWriter
import clip
import torch.nn.functional as F
from itertools import cycle
from model.backbones.tokenization_bert import BertTokenizer
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import cv2
import pickle
import json


def mkdir_if_missing(dir_path):
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(os.path.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(state, is_best, fpath):
    if len(fpath) != 0:
        mkdir_if_missing(fpath)

    fpath = os.path.join(fpath, 'checkpoint.pth')
    torch.save(state, fpath, _use_new_zipfile_serialization=False)
    if is_best:
        shutil.copy(fpath, os.path.join(os.path.dirname(fpath), 'checkpoint_best.pth'))
    print('save model at ', fpath)



def get_phrase_level_txt(caption):
    phrase_b = []
    for cap in caption:
        cap_list = cap.strip('.').strip('[').strip(']').split(".")
        result = []
        for cap in cap_list:
            caps = cap.split(',')
            nums = [len(item) for item in caps]
            idx = np.argmax(nums)
            item_1 = caps[idx]
            item_2 = (",".join(caps[:idx]) + "," + ",".join(caps[idx + 1:])).strip(",")
            item = [item_1, item_2]
            item = [it for it in item if len(it) > 5]
            result += item
        result = [it.strip(" ") for it in result if len(it) > 0]
        phrase = random.sample(result, 1)
        phrase_b.append(phrase[0])
    return phrase_b





def do_train_cuhkpedes_uvt_level_data(cfg, model, train_loader, test_loader, optimizer, scheduler, loss_func, loss_cmpm, tokenizer, local_rank, start_epoch, logger):
    log_period = int(len(train_loader) / 10)  # cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    scaler = amp.GradScaler()
    last_acc_val = 0.0

    # train
    for epoch in range(1, epochs + 1):  # 120
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        scheduler.step(epoch)
        model.train()
        num_iter = len(train_loader)
        for n_iter, input in enumerate(train_loader):  # [64, 3, 256, 128], [64,], [64,], [64,]
            img1, img2, img3, text, pid, pathes, word = input
            img = torch.cat([img1, img2, img3], dim=0)

            optimizer.zero_grad()
            img = img.to(device)        # [4, 3, 256, 128]
            pid = pid.to(device)        # [4, ]
            batch = img1.shape[0]

            phrase = get_phrase_level_txt(text)
            text_3 = list(word) + list(phrase) + list(text)
            pid_3 = torch.cat([pid, pid, pid])

            text_dict = tokenizer(text_3, padding='longest', max_length=100, return_tensors="pt").to(device)
            text_token = text_dict.data['input_ids']
            mask = text_dict.data['attention_mask']

            # single-precison training
            with amp.autocast(enabled=True):
                out_vis, out_txt = model(img, text_token, mask)        # [12, 768], [12, 768]

                # loss
                loss = loss_cmpm(out_vis, out_txt, pid_3)     # 191.6707

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # ## double-precison training
            # out_vis, out_txt = model(vis, text_token, mask)  # [4, 11003], [4, 2048]
            #
            # # loss
            # loss = loss_cmpm(out_vis, out_txt, pid)  # 191.6707
            #
            # loss.backward()
            # optimizer.step()

            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if n_iter % log_period == 0:  # 50
                line = "Epoch[%d] Iteration[%d/%d] \tloss: %.2f, lr: %.2f" % \
                       (epoch, n_iter + 1, num_iter, loss.item(), scheduler._get_lr(epoch)[0])
                logger.info(line)

            torch.cuda.empty_cache()

        end_time = time.time()
        time_per_batch = (end_time - start_time) / num_iter
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        # Evalution
        eval_period = 1
        if epoch % eval_period == 0:  # eval_period=120
            print('Test:', epoch)
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    R1, R5, R10, mAP = do_inference_cuhk_uvt_base(cfg, model, test_loader, logger, tokenizer)
                    torch.cuda.empty_cache()

                    acc_test = 0.5 * (R1 + mAP)
                    is_best = acc_test >= last_acc_val
                    save_checkpoint({
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch + 1,
                        'best_acc': acc_test,
                    }, is_best, fpath=cfg.OUTPUT_DIR)
                    if is_best:
                        last_acc_val = acc_test

                    writer.add_scalar('train_loss', float(loss_meter.avg), epoch + 1)
                    writer.add_scalar('R1', float(R1), epoch + 1)
                    writer.add_scalar('R5', float(R5), epoch + 1)
                    writer.add_scalar('mAP', float(mAP), epoch + 1)
            else:
                R1, R5, R10, mAP = do_inference_cuhk_uvt_base(cfg, model, test_loader, logger, tokenizer)
                torch.cuda.empty_cache()

                acc_test = 0.5 * (R1 + mAP)
                is_best = acc_test >= last_acc_val
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'best_acc': acc_test,
                }, is_best, fpath=cfg.OUTPUT_DIR)
                if is_best:
                    last_acc_val = acc_test

                lr = scheduler._get_lr(epoch)[0]
                writer.add_scalar('train_loss', float(loss_meter.avg), epoch + 1)
                writer.add_scalar('lr', float(lr), epoch + 1)
                writer.add_scalar('R1', float(R1), epoch + 1)
                writer.add_scalar('R5', float(R5), epoch + 1)
                writer.add_scalar('mAP', float(mAP), epoch + 1)

    # Test
    print('Final Test:')
    last_model_wts = torch.load(os.path.join(cfg.OUTPUT_DIR, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    cmc_i2i, mAP_i2i, cmc_t2i, mAP_t2i = do_inference_cuhk_uvt_base(cfg, model, test_loader, logger, tokenizer)





def do_train_icfg_uvt_level_data(cfg, model, train_loader, test_loader, optimizer, scheduler, loss_func, loss_cmpm, tokenizer, local_rank, start_epoch, logger):
    log_period = int(len(train_loader) / 10)  # cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    scaler = amp.GradScaler()
    last_acc_val = 0.0

    # train
    for epoch in range(start_epoch, epochs + 1):  # 120
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        scheduler.step(epoch)
        model.train()
        num_iter = len(train_loader)
        for n_iter, input in enumerate(train_loader):  # [64, 3, 256, 128], [64,], [64,], [64,]
            img1, img2, img3, text, pid, pathes, word = input
            img = torch.cat([img1, img2, img3], dim=0)

            optimizer.zero_grad()
            img = img.to(device)        # [4, 3, 256, 128]
            pid = pid.to(device)        # [4, ]
            batch = img1.shape[0]

            phrase = get_phrase_level_txt(text)
            text_3 = list(word) + list(phrase) + list(text)
            pid_3 = torch.cat([pid, pid, pid])

            text_dict = tokenizer(text_3, padding='longest', max_length=100, return_tensors="pt").to(device)
            text_token = text_dict.data['input_ids']
            mask = text_dict.data['attention_mask']

            # single-precison training
            with amp.autocast(enabled=True):
                out_vis, out_txt = model(img, text_token, mask)         # [12, 768], [12, 768]
                # loss
                loss = loss_cmpm(out_vis, out_txt, pid_3)               # 7.2126

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # ## double-precison training
            # out_vis, out_txt = model(vis, text_token, mask)  # [4, 11003], [4, 2048]
            #
            # # loss
            # loss = loss_cmpm(out_vis, out_txt, pid)  # 191.6707
            #
            # loss.backward()
            # optimizer.step()

            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if n_iter % log_period == 0:  # 50
                line = "Epoch[%d] Iteration[%d/%d] \tloss: %.2f, lr: %.2f" % \
                       (epoch, n_iter + 1, num_iter, loss.item(), scheduler._get_lr(epoch)[0])
                if cfg.MODEL.DIST_TRAIN:
                    if dist.get_rank() == 0:
                        logger.info(line)
                else:
                    logger.info(line)

            torch.cuda.empty_cache()

        end_time = time.time()
        time_per_batch = (end_time - start_time) / num_iter
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        # Evalution
        eval_period = 1
        if epoch % eval_period == 0:  # eval_period=120
            print('Test:', epoch)
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    R1, R5, R10, mAP = do_inference_icfg_base(cfg, model, test_loader, logger, tokenizer)
                    torch.cuda.empty_cache()

                    acc_test = 0.5 * (R1 + mAP)
                    is_best = acc_test >= last_acc_val
                    save_checkpoint({
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch + 1,
                        'best_acc': acc_test,
                    }, is_best, fpath=cfg.OUTPUT_DIR)
                    if is_best:
                        last_acc_val = acc_test

                    # lr = scheduler.get_last_lr()[0]         # scheduler._get_lr(epoch)[0]
                    writer.add_scalar('train_loss', float(loss_meter.avg), epoch + 1)
                    # writer.add_scalar('lr', float(lr), epoch + 1)
                    writer.add_scalar('R1', float(R1), epoch + 1)
                    writer.add_scalar('R5', float(R5), epoch + 1)
                    writer.add_scalar('mAP', float(mAP), epoch + 1)
            else:
                R1, R5, R10, mAP = do_inference_icfg_base(cfg, model, test_loader, logger, tokenizer)
                torch.cuda.empty_cache()

                acc_test = 0.5 * (R1 + mAP)
                is_best = acc_test >= last_acc_val
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'best_acc': acc_test,
                }, is_best, fpath=cfg.OUTPUT_DIR)
                if is_best:
                    last_acc_val = acc_test

                lr = scheduler._get_lr(epoch)[0]
                writer.add_scalar('train_loss', float(loss_meter.avg), epoch + 1)
                writer.add_scalar('lr', float(lr), epoch + 1)
                writer.add_scalar('R1', float(R1), epoch + 1)
                writer.add_scalar('R5', float(R5), epoch + 1)
                writer.add_scalar('mAP', float(mAP), epoch + 1)

    # Test
    print('Final Test:')
    last_model_wts = torch.load(os.path.join(cfg.OUTPUT_DIR, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    cmc_i2i, mAP_i2i, cmc_t2i, mAP_t2i = do_inference_icfg_base(cfg, model, test_loader, logger, tokenizer)







def do_train(cfg, model, center_criterion, train_loader, val_loader, optimizer, optimizer_center, scheduler, loss_fn, num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):      # 120
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):     # [64, 3, 256, 128], [64,], [64,], [64,]
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view)      # [64, 702], [64, 768]
                loss = loss_fn(score, feat, target, target_cam)         # 11.5966

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:      # checkpoint_period=120
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:        # eval_period=120
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()





def do_train_rstp_uvt_level_data(cfg, model, train_loader, test_loader, optimizer, scheduler, loss_func, loss_cmpm, tokenizer, local_rank, start_epoch, logger):
    log_period = int(len(train_loader) / 10)  # cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    scaler = amp.GradScaler()
    last_acc_val = 0.0

    # train
    for epoch in range(start_epoch, epochs + 1):  # 120
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        scheduler.step(epoch)
        model.train()
        num_iter = len(train_loader)
        for n_iter, input in enumerate(train_loader):  # [64, 3, 256, 128], [64,], [64,], [64,]
            img1, img2, img3, text, pid, pathes, word = input
            img = torch.cat([img1, img2, img3], dim=0)

            optimizer.zero_grad()
            img = img.to(device)        # [4, 3, 256, 128]
            pid = pid.to(device)        # [4, ]
            batch = img1.shape[0]

            phrase = get_phrase_level_txt(text)
            text_3 = list(word) + list(phrase) + list(text)
            pid_3 = torch.cat([pid, pid, pid])

            text_dict = tokenizer(text_3, padding='longest', max_length=100, return_tensors="pt").to(device)
            text_token = text_dict.data['input_ids']
            mask = text_dict.data['attention_mask']

            # single-precison training
            with amp.autocast(enabled=True):
                out_vis, out_txt = model(img, text_token, mask)         # [12, 768], [12, 768]

                # loss
                loss = loss_cmpm(out_vis, out_txt, pid_3)               # 7.2126

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # ## double-precison training
            # out_vis, out_txt = model(vis, text_token, mask)  # [4, 11003], [4, 2048]
            #
            # # loss
            # loss = loss_cmpm(out_vis, out_txt, pid)  # 191.6707
            #
            # loss.backward()
            # optimizer.step()

            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if n_iter % log_period == 0:  # 50
                line = "Epoch[%d] Iteration[%d/%d] \tloss: %.2f, lr: %.2f" % \
                       (epoch, n_iter + 1, num_iter, loss.item(), scheduler._get_lr(epoch)[0])
                logger.info(line)

            torch.cuda.empty_cache()

        end_time = time.time()
        time_per_batch = (end_time - start_time) / num_iter
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        # Evalution
        eval_period = 1
        if epoch % eval_period == 0:  # eval_period=120
            print('Test:', epoch)
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    R1, R5, R10, mAP = do_inference_rstp_uvt_base(cfg, model, test_loader, logger, tokenizer)
                    torch.cuda.empty_cache()

                    acc_test = 0.5 * (R1 + mAP)
                    is_best = acc_test >= last_acc_val
                    save_checkpoint({
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch + 1,
                        'best_acc': acc_test,
                    }, is_best, fpath=cfg.OUTPUT_DIR)
                    if is_best:
                        last_acc_val = acc_test

                    # lr = scheduler.get_last_lr()[0]         # scheduler._get_lr(epoch)[0]
                    writer.add_scalar('train_loss', float(loss_meter.avg), epoch + 1)
                    # writer.add_scalar('lr', float(lr), epoch + 1)
                    writer.add_scalar('R1', float(R1), epoch + 1)
                    writer.add_scalar('R5', float(R5), epoch + 1)
                    writer.add_scalar('mAP', float(mAP), epoch + 1)
            else:
                R1, R5, R10, mAP = do_inference_rstp_uvt_base(cfg, model, test_loader, logger, tokenizer)
                torch.cuda.empty_cache()

                acc_test = 0.5 * (R1 + mAP)
                is_best = acc_test >= last_acc_val
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'best_acc': acc_test,
                }, is_best, fpath=cfg.OUTPUT_DIR)
                if is_best:
                    last_acc_val = acc_test

                lr = scheduler._get_lr(epoch)[0]
                writer.add_scalar('train_loss', float(loss_meter.avg), epoch + 1)
                writer.add_scalar('lr', float(lr), epoch + 1)
                writer.add_scalar('R1', float(R1), epoch + 1)
                writer.add_scalar('R5', float(R5), epoch + 1)
                writer.add_scalar('mAP', float(mAP), epoch + 1)

    # Test
    print('Final Test:')
    last_model_wts = torch.load(os.path.join(cfg.OUTPUT_DIR, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    cmc_i2i, mAP_i2i, cmc_t2i, mAP_t2i = do_inference_rstp_uvt_base(cfg, model, test_loader, logger, tokenizer)



def do_train_cuhk_img_pre(cfg, model, train_loader, test_loader, optimizer, scheduler, loss_func, loss_cmpm, tokenizer, local_rank, start_epoch, logger):
    log_period = int(len(train_loader) / 10)  # cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    # logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter_itc = AverageMeter()
    loss_meter_itm = AverageMeter()
    loss_meter_mlm = AverageMeter()

    scaler = amp.GradScaler()
    last_acc_val = 0.0

    # train
    for epoch in range(start_epoch, epochs + 1):  # 120
        start_time = time.time()
        loss_meter_itc.reset()
        loss_meter_itm.reset()
        loss_meter_mlm.reset()
        scheduler.step(epoch)
        model.train()
        num_iter = len(train_loader)
        for n_iter, input in enumerate(train_loader):  # [64, 3, 256, 128], [64,], [64,], [64,]
            img, text, pid, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, paths = input

            optimizer.zero_grad()
            img = img.to(device)      # [4, 3, 256, 128]
            pid = pid.to(device)        # [4, ]
            text_ids = text_ids.to(device)
            text_atts = text_atts.to(device)
            text_ids_masked = text_ids_masked.to(device)
            masked_pos = masked_pos.to(device)
            masked_ids = masked_ids.to(device)
            batch = img.shape[0]


            # # create empty image-text pairs
            # text_dict = tokenizer(text, padding='longest', max_length=100, return_tensors="pt").to(device)
            # text_token = text_dict.data['input_ids']
            # mask = text_dict.data['attention_mask']

            ## single-precison training
            with amp.autocast(enabled=True):
                loss_itc, loss_itm, loss_mlm = model(img, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, pid)   # [4, 2048], [4, 11003]

                # loss
                loss = loss_itc + loss_itm + loss_mlm

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter_itc.update(loss_itc.item(), img.shape[0])
            loss_meter_itm.update(loss_itm.item(), img.shape[0])
            loss_meter_mlm.update(loss_mlm.item(), img.shape[0])

            torch.cuda.synchronize()
            if n_iter % log_period == 0:  # 50
                line = "Epoch[%d] Iteration[%d/%d] \tloss_itc: %.2f, loss_itm: %.2f, loss_mlm: %.2f, lr: %.2f" % \
                       (epoch, n_iter + 1, num_iter, loss_itc.item(), loss_itm.item(), loss_mlm.item(), scheduler._get_lr(epoch)[0])
                if cfg.MODEL.DIST_TRAIN:
                    if dist.get_rank() == 0:
                        logger.info(line)
                else:
                    logger.info(line)

            torch.cuda.empty_cache()

        end_time = time.time()
        time_per_batch = (end_time - start_time) / num_iter
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        # Evalution
        eval_period = 1
        if epoch % eval_period == 0:  # eval_period=120
            print('Test:', epoch)
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    R1, R5, R10, mAP = do_inference_cuhk_img_txt_pre(cfg, model, test_loader, logger, tokenizer)
                    torch.cuda.empty_cache()

                    acc_test = 0.5 * (R1 + mAP)
                    is_best = acc_test >= last_acc_val
                    save_checkpoint({
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch + 1,
                        'best_acc': acc_test,
                    }, is_best, fpath=cfg.OUTPUT_DIR)
                    if is_best:
                        last_acc_val = acc_test

                    # lr = scheduler.get_last_lr()[0]         # scheduler._get_lr(epoch)[0]
                    writer.add_scalar('loss_itc', float(loss_meter_itc.avg), epoch + 1)
                    writer.add_scalar('loss_itm', float(loss_meter_itm.avg), epoch + 1)
                    writer.add_scalar('loss_itm', float(loss_meter_mlm.avg), epoch + 1)
                    # writer.add_scalar('lr', float(lr), epoch + 1)
                    writer.add_scalar('R1', float(R1), epoch + 1)
                    writer.add_scalar('mAP', float(mAP), epoch + 1)
            else:
                R1, R5, R10, mAP = do_inference_cuhk_img_txt_pre(cfg, model, test_loader, logger, tokenizer)
                torch.cuda.empty_cache()

                acc_test = 0.5 * (R1 + mAP)
                is_best = acc_test >= last_acc_val
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'best_acc': acc_test,
                }, is_best, fpath=cfg.OUTPUT_DIR)
                if is_best:
                    last_acc_val = acc_test

                lr = scheduler._get_lr(epoch)[0]
                # writer.add_scalar('train_loss', float(loss), epoch + 1)
                writer.add_scalar('lr', float(lr), epoch + 1)
                writer.add_scalar('loss_itc', float(loss_meter_itc.avg), epoch + 1)
                writer.add_scalar('loss_itm', float(loss_meter_itm.avg), epoch + 1)
                writer.add_scalar('loss_itm', float(loss_meter_mlm.avg), epoch + 1)
                writer.add_scalar('R1', float(R1), epoch + 1)
                writer.add_scalar('mAP', float(mAP), epoch + 1)

    # Test
    print('Final Test:')
    last_model_wts = torch.load(os.path.join(cfg.OUTPUT_DIR, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    cmc_i2i, mAP_i2i, cmc_t2i, mAP_t2i = do_inference_cuhk_img_txt_pre(cfg, model, test_loader, logger, tokenizer)


def do_inference_cuhk_uvt_base(cfg, model, val_loader, logger=None, tokenizer=None):
    device = "cuda"
    if logger is not None:
        logger.info('Test:')

    evaluator = R1_mAP_CUHK(max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    if device:
        model.to(device)

    model.eval()
    with torch.no_grad():
        for n_iter, input in enumerate(val_loader):     # len(val_loader)=97
            img, text, pid, paths, token = input
            img = img.to(device)                            # [4, 3, 256, 128]

            text_dict = tokenizer(text, padding='longest', max_length=100, return_tensors="pt").to(device)
            text_token = text_dict.data['input_ids']
            mask = text_dict.data['attention_mask']

            out_vis, out_txt = model(img, text_token, mask)  # [4, 768], [4, 768]

            target = pid.data.numpy().tolist()
            evaluator.update((out_vis, out_txt, target))

        cmc, mAP = evaluator.compute()
        line_t2i = 'text2img: \t rank1: %.4f, rank5: %.4f, rank10: %.4f, rank20: %.4f, mAP: %.4f' % (cmc[0], cmc[4], cmc[9], cmc[19], mAP)
        logger.info(line_t2i)

    return cmc[0], cmc[4], cmc[9], mAP




def do_inference_cuhk_bert(cfg, model, val_loader, logger=None, tokenizer=None):
    device = "cuda"
    if logger is not None:
        logger.info('Test:')

    evaluator = R1_mAP_CUHK(max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    if device:
        model.to(device)

    model.eval()
    with torch.no_grad():
        for n_iter, input in enumerate(val_loader):     # len(val_loader)=97
            img, text, pid, paths, token = input
            vis = copy.deepcopy(img).detach()
            img = img.to(device)                            # [4, 3, 256, 128]
            vis = vis.to(device)  # [4, 3, 256, 128]
            # text_token, mask = clip.tokenize_mask(text, context_length=64)  # [4, 64]
            # text_token = text_token.to(device)  # [4, 64]
            # mask = mask.to(device)  # [4, 64], 0, 1

            text_dict = tokenizer(text, padding='longest', max_length=100, return_tensors="pt").to(device)
            text_token = text_dict.data['input_ids']
            mask = text_dict.data['attention_mask']

            img_score, img_feat, out_vis, out_txt = model(img, vis, text_token, mask)  # [4, 768], [4, 768]

            target = pid.data.numpy().tolist()
            evaluator.update((out_vis, out_txt, target))

        cmc, mAP = evaluator.compute()
        line_t2i = 'text2img: \t rank1: %.4f, rank5: %.4f, rank10: %.4f, rank20: %.4f, mAP: %.4f' % (cmc[0], cmc[4], cmc[9], cmc[19], mAP)
        logger.info(line_t2i)

    return cmc[0], cmc[4], cmc[9], mAP



def do_inference_icfg_base(cfg, model, val_loader, logger=None, tokenizer=None):
    device = "cuda"
    if logger is not None:
        logger.info('Test:')

    evaluator = R1_mAP_ICFG(max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    if device:
        model.to(device)

    model.eval()
    with torch.no_grad():
        for n_iter, input in enumerate(val_loader):     # len(val_loader)=97
            img, text, pid, paths, token = input
            img = img.to(device)                            # [4, 3, 256, 128]

            text_dict = tokenizer(text, padding='longest', max_length=100, return_tensors="pt").to(device)
            text_token = text_dict.data['input_ids']
            mask = text_dict.data['attention_mask']

            out_vis, out_txt = model(img, text_token, mask)  # [4, 768], [4, 768]

            target = pid.data.numpy().tolist()
            evaluator.update((out_vis, out_txt, target))

        cmc, mAP = evaluator.compute()
        line_t2i = 'text2img: \t rank1: %.4f, rank5: %.4f, rank10: %.4f, rank20: %.4f, mAP: %.4f' % (cmc[0], cmc[4], cmc[9], cmc[19], mAP)
        logger.info(line_t2i)

    return cmc[0], cmc[4], cmc[9], mAP


def do_inference_rstp_uvt_base(cfg, model, val_loader, logger=None, tokenizer=None):
    device = "cuda"
    if logger is not None:
        logger.info('Test:')

    evaluator = R1_mAP_CUHK(max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    if device:
        model.to(device)

    model.eval()
    with torch.no_grad():
        for n_iter, input in enumerate(val_loader):     # len(val_loader)=97
            img, text, pid, paths, token = input
            img = img.to(device)                            # [4, 3, 256, 128]

            text_dict = tokenizer(text, padding='longest', max_length=100, return_tensors="pt").to(device)
            text_token = text_dict.data['input_ids']
            mask = text_dict.data['attention_mask']

            out_vis, out_txt = model(img, text_token, mask)  # [4, 768], [4, 768]

            target = pid.data.numpy().tolist()
            evaluator.update((out_vis, out_txt, target))

        cmc, mAP = evaluator.compute()
        line_t2i = 'text2img: \t rank1: %.4f, rank5: %.4f, rank10: %.4f, rank20: %.4f, mAP: %.4f' % (cmc[0], cmc[4], cmc[9], cmc[19], mAP)
        logger.info(line_t2i)

    return cmc[0], cmc[4], cmc[9], mAP




def do_inference_cuhk_img_txt_pre(cfg, model, val_loader, logger=None, tokenizer=None):
    device = "cuda"
    if logger is not None:
        logger.info('Test:')

    evaluator = R1_mAP_CUHK(max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    if device:
        model.to(device)

    model.eval()
    with torch.no_grad():
        for n_iter, input in enumerate(val_loader):     # len(val_loader)=97
            img, text, pid, paths = input
            img = img.to(device)                            # [4, 3, 256, 128]
            # text_token, mask = clip.tokenize_mask(text, context_length=64)  # [4, 64]
            # text_token = text_token.to(device)  # [4, 64]
            # mask = mask.to(device)  # [4, 64], 0, 1

            text_dict = tokenizer(text, padding='longest', max_length=100, return_tensors="pt").to(device)
            text_token = text_dict.data['input_ids']
            mask = text_dict.data['attention_mask']

            out_vis, out_txt = model(img, text_token, mask)  # [4, 768], [4, 768]

            target = pid.data.numpy().tolist()
            evaluator.update((out_vis, out_txt, target))

        cmc, mAP = evaluator.compute()
        line_t2i = 'text2img: \t rank1: %.4f, rank5: %.4f, rank10: %.4f, rank20: %.4f, mAP: %.4f' % (cmc[0], cmc[4], cmc[9], cmc[19], mAP)
        logger.info(line_t2i)

    return cmc[0], cmc[4], cmc[9], mAP










