import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
import os
import math
import numpy as np
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from .backbones.vit_vl import vit_base_patch16_224_uvt_img_txt, vit_base_patch16_224_uvt_img_txt_mask

from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss

from model.backbones.xbert import BertConfig, BertModel, BertForMaskedLM

from model.backbones.xbert_fusion import BertForMaskedLM as BertForMaskedLM_F
from model.backbones.modeling_bert import BertForMaskedLM as BertForMaskedLM_ori

from model.backbones.xbert_fusion_vl import BertModel as BertModelVL
from model.backbones.xbert_fusion_vl import BertForMaskedLM as BertForMaskedLM_VL
from model.backbones.modeling_bert_vl import BertForMaskedLM as BertForMaskedLM_ori_VL
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from model.backbones.clip_transformer import Transformer
from model.backbones.bert_transformer import FusionTransformer
from model.backbones.detr import build_detr, build_VLFusion
import clip
# from pytorch_pretrained_bert.modeling import BertModel
from torch.cuda.amp import autocast as autocast
from model.backbones.vit_vl import PatchEmbed_overlap
import transformers as ppb
import torch.distributed as dist


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )


allgather = AllGather.apply

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def shuffle_unit(features, shift, group, begin=1):
    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin - 1 + shift:], features[:, begin:begin - 1 + shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            # self.base = ResNet(last_stride=last_stride,
            #                    block=Bottleneck,
            #                    layers=[3, 4, 6, 3])
            self.base = ResNet(last_stride=last_stride,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate=cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        if pretrain_choice == 'coco':
            self.base.load_param_ALBEF(model_path)
            print('Loading pretrained coco model......from {}'.format(model_path))

        if pretrain_choice == 'flickr':
            self.base.load_param_ALBEF(model_path)
            print('Loading pretrained flickr model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)  # [64, 768]

        feat = self.bottleneck(global_feat)  # [64, 768]

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)  # [64, 702]

            return cls_score, global_feat  # [64, 702], [64, 768], global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))



class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes  # 702
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE  # 'softmax'
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)  # 768->702
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)  # 768->702
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)  # 768->702
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)  # 768->702
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)  # 768->702
            self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP  # 2
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM  # 5
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH  # 4
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, x, label=None, cam_label=None, view_label=None):  # label is unused if self.cos_layer == 'no'
        # x->[64, 3, 256, 128]
        features = self.base(x, cam_label=cam_label, view_label=view_label)  # [64, 129, 768]

        # global branch
        b1_feat = self.b1(features)  # [64, 129, 768]
        global_feat = b1_feat[:, 0]  # [64, 768]

        # JPM branch
        feature_length = features.size(1) - 1  # 128
        patch_length = feature_length // self.divide_length  # 32
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length * 2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length * 2:patch_length * 3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length * 3:patch_length * 4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3, cls_score_4], \
                   [global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def conv1x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), stride=stride,
                     padding=(0, 1), groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv1x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):  # [4, 1024, 1, 64]
        identity = x

        out = self.conv1(x)  # [4, 512, 1, 64]
        out = self.bn1(out)  # [4, 512, 1, 64]
        out = self.relu(out)  # [4, 512, 1, 64]

        out = self.conv2(out)  # [4, 512, 1, 64]
        out = self.bn2(out)  # [4, 512, 1, 64]
        out = self.relu(out)  # [4, 512, 1, 64]

        out = self.conv3(out)  # [4, 2048, 1, 64]
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # [4, 2048, 1, 64]

        out += identity  # [4, 2048, 1, 64]
        out = self.relu(out)

        return out  # [4, 2048, 1, 64]


class ResNetTXT(nn.Module):

    def __init__(self, inplanes=768, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNetTXT, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = inplanes

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.conv1 = conv1x1(self.inplanes, self.inplanes)  # 768 -> 1024
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        downsample = nn.Sequential(
            conv1x1(self.inplanes, self.inplanes),
            norm_layer(self.inplanes),
        )

        # 3, 4, 6, 3
        self.branch1 = nn.Sequential(
            Bottleneck(inplanes=self.inplanes, planes=self.inplanes, width=512, downsample=downsample),
            Bottleneck(inplanes=self.inplanes, planes=self.inplanes, width=512),
            Bottleneck(inplanes=self.inplanes, planes=self.inplanes, width=512)
        )
        self.branch2 = nn.Sequential(
            Bottleneck(inplanes=self.inplanes, planes=self.inplanes, width=512, downsample=downsample),
            Bottleneck(inplanes=self.inplanes, planes=self.inplanes, width=512),
            Bottleneck(inplanes=self.inplanes, planes=self.inplanes, width=512)
        )
        self.branch3 = nn.Sequential(
            Bottleneck(inplanes=self.inplanes, planes=self.inplanes, width=512, downsample=downsample),
            Bottleneck(inplanes=self.inplanes, planes=self.inplanes, width=512),
            Bottleneck(inplanes=self.inplanes, planes=self.inplanes, width=512)
        )
        self.branch4 = nn.Sequential(
            Bottleneck(inplanes=self.inplanes, planes=self.inplanes, width=512, downsample=downsample),
            Bottleneck(inplanes=self.inplanes, planes=self.inplanes, width=512),
            Bottleneck(inplanes=self.inplanes, planes=self.inplanes, width=512)
        )
        self.branch5 = nn.Sequential(
            Bottleneck(inplanes=self.inplanes, planes=self.inplanes, width=512, downsample=downsample),
            Bottleneck(inplanes=self.inplanes, planes=self.inplanes, width=512),
            Bottleneck(inplanes=self.inplanes, planes=self.inplanes, width=512)
        )
        self.branch6 = nn.Sequential(
            Bottleneck(inplanes=self.inplanes, planes=self.inplanes, width=512, downsample=downsample),
            Bottleneck(inplanes=self.inplanes, planes=self.inplanes, width=512),
            Bottleneck(inplanes=self.inplanes, planes=self.inplanes, width=512)
        )

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.norm1 = nn.LayerNorm(self.inplanes)
        self.norm2 = nn.LayerNorm(self.inplanes)
        self.norm3 = nn.LayerNorm(self.inplanes)
        self.norm4 = nn.LayerNorm(self.inplanes)
        self.norm5 = nn.LayerNorm(self.inplanes)
        self.norm6 = nn.LayerNorm(self.inplanes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):               # [4, 768, 128, 1]
        x1 = self.conv1(x)              # [64, 768, 64, 1]
        x1 = self.bn1(x1)
        x1 = self.relu(x1)              # [64, 768, 64, 1]
        x21 = self.branch1(x1)          # [64, 768, 64, 1]
        x22 = self.branch2(x1)          # [64, 768, 64, 1]
        x23 = self.branch3(x1)          # [64, 768, 64, 1]
        x24 = self.branch4(x1)          # [64, 768, 64, 1]
        x25 = self.branch5(x1)          # [64, 768, 64, 1]
        x26 = self.branch6(x1)          # [64, 768, 64, 1]

        x21 = self.max_pool(x21).squeeze(dim=-1).squeeze(dim=-1)        # [64, 768]
        x22 = self.max_pool(x22).squeeze(dim=-1).squeeze(dim=-1)        # [64, 768]
        x23 = self.max_pool(x23).squeeze(dim=-1).squeeze(dim=-1)        # [64, 768]
        x24 = self.max_pool(x24).squeeze(dim=-1).squeeze(dim=-1)        # [64, 768]
        x25 = self.max_pool(x25).squeeze(dim=-1).squeeze(dim=-1)        # [64, 768]
        x26 = self.max_pool(x26).squeeze(dim=-1).squeeze(dim=-1)        # [64, 768]

        x21 = self.norm1(x21)
        x22 = self.norm2(x22)
        x23 = self.norm3(x23)
        x24 = self.norm4(x24)
        x25 = self.norm5(x25)
        x26 = self.norm6(x26)

        return x21, x22, x23, x24, x25, x26     # [64, 2048], ..., [64, 2048]



class ResNetVIS(nn.Module):
    def __init__(self, inplanes=768, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNetVIS, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = inplanes

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.conv1 = conv1x1(self.inplanes, self.inplanes)  # 768 -> 1024
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        downsample = nn.Sequential(
            conv1x1(self.inplanes, self.inplanes),
            norm_layer(self.inplanes),
        )

        # 3, 4, 6, 3
        self.branch = nn.Sequential(
            Bottleneck(inplanes=self.inplanes, planes=self.inplanes, width=512, downsample=downsample),
            Bottleneck(inplanes=self.inplanes, planes=self.inplanes, width=512),
            Bottleneck(inplanes=self.inplanes, planes=self.inplanes, width=512)
        )
        self.split_pool = nn.AdaptiveMaxPool2d((1, 6))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):           # [64, 768, 1, 128]
        x1 = self.conv1(x)          # [64, 1024, 1, 128]
        x1 = self.bn1(x1)
        x1 = self.relu(x1)          # [64, 1024, 1, 128]

        x2 = self.branch(x1)        # [64, 2048, 1, 128]
        xs = self.split_pool(x2)    # [64, 2048, 1, 6]
        x21 = xs[:, :, 0, 0]        # [64, 2048]
        x22 = xs[:, :, 0, 1]        # [64, 2048]
        x23 = xs[:, :, 0, 2]        # [64, 2048]
        x24 = xs[:, :, 0, 3]        # [64, 2048]
        x25 = xs[:, :, 0, 4]        # [64, 2048]
        x26 = xs[:, :, 0, 5]        # [64, 2048]

        x2 = self.max_pool(x2).squeeze(dim=-1).squeeze(dim=-1)      # [64, 2048]
        x1 = self.max_pool(x1).squeeze(dim=-1).squeeze(dim=-1)      # [64, 1024]

        return x1, x2, x21, x22, x23, x24, x25, x26  # [64, 1024], [64, 2048], [64, 2048], ..., [64, 2048]


class ResNetIMG(nn.Module):
    def __init__(self, inplanes=768, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNetIMG, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = inplanes

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.conv1 = conv1x1(self.inplanes, 1024)  # 768 -> 1024
        self.bn1 = norm_layer(1024)
        self.relu = nn.ReLU(inplace=True)

        downsample = nn.Sequential(
            conv1x1(1024, 2048),
            norm_layer(2048),
        )

        # 3, 4, 6, 3
        self.branch = nn.Sequential(
            Bottleneck(inplanes=1024, planes=2048, width=512, downsample=downsample),
            Bottleneck(inplanes=2048, planes=2048, width=512),
            Bottleneck(inplanes=2048, planes=2048, width=512)
        )

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):           # [4, 768, 1, 128]
        x1 = self.conv1(x)          # [4, 1024, 1, 128]
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = self.branch(x1)        # [4, 2048, 1, 128]

        x2 = self.max_pool(x2).squeeze(dim=-1).squeeze(dim=-1)

        return x2  # [4, 2048]




class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        self.config = config

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):  # [2, 11]
        if input_ids is not None:
            input_shape = input_ids.size()  # (2, 11)
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]  # 11

        if position_ids is None:  # self.position_ids -> [1, 512]
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]  # [1, 11]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)  # [2, 11, 768]

        token_type_embeddings = self.token_type_embeddings(token_type_ids)  # [2, 11, 768] <- [2, 11]

        embeddings = inputs_embeds + token_type_embeddings  # [2, 11, 768]
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)  # [1, 11, 768]
            embeddings += position_embeddings  # [2, 11, 768]
        embeddings = self.LayerNorm(embeddings)  # [2, 11, 768]
        embeddings = self.dropout(embeddings)  # [2, 11, 768]
        return embeddings  # [2, 11, 768]




class build_transformer_uvt_img_txt_mask(nn.Module):
    def __init__(self, num_classes, cfg, factory, context_length=64):
        super(build_transformer_uvt_img_txt_mask, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        bert_path = cfg.MODEL.BERT_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.feat_dim = 768
        self.use_mask = cfg.MODEL.USE_MASK
        # temp 默认0.07
        self.temp = nn.Parameter(torch.ones([]) * cfg.MODEL.TEMP)
        self.context_length = context_length

        ## base
        bert_config = BertConfig.from_json_file(cfg.TEXT.BERT_CONFIG)
        text_encoder = BertModel.from_pretrained(cfg.TEXT.TEXT_ENCODER, config=bert_config, add_pooling_layer=False)

        self.encoder_vl = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                              drop_rate=cfg.MODEL.DROP_OUT, attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, config=bert_config)

        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.encoder_vl.load_param_hybrid(model_path, text_encoder)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes  # 11003
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.classifier_img = nn.Linear(self.feat_dim, self.num_classes, bias=False)  # 2048 -> 11003
        self.classifier_img.apply(weights_init_classifier)

        self.bottleneck_img = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck_img.bias.requires_grad_(False)
        self.bottleneck_img.apply(weights_init_kaiming)


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    def encode_text(self, text):  # [24, 77]
        x = self.token_embedding(text).type(self.dtype)  # [24, 77, 512]
        x = x + self.positional_embedding.type(self.dtype).expand_as(x)  # [24, 77, 512]

        x = x.permute(1, 0, 2)  # [77, 24, 512]
        x = self.text_encoder(x)  # [77, 24, 512]
        x = x.permute(1, 0, 2)  # [24, 77, 512]
        x = self.ln_final(x).type(self.dtype)  # [24, 77, 512]

        return x  # [24, 77, 512]

    def embed_position(self, x, mode='img'):        # [4, 64]
        if mode == 'txt':
            x = self.token_embedding(x).type(self.dtype)    # [4, 64, 768]
            x = x + self.pos_embed_txt.type(self.dtype).expand_as(x)  # [24, 64, 768]
        else:
            x = self.patch_embed(x)     # [64, 128, 768]
            x = x + self.pos_embed
        return x

    def forward(self, vis, txt, mask, use_mask=True, percent=0.2):  # [24, 3, 256, 128]
        if not self.training:
            use_mask = False

        # vis
        t_vis = self.encoder_vl(vis, mode='vis', use_mask=use_mask, percent=percent)        # [4, 128, 768]
        out_vis = t_vis[:, 0]                           # [4, 768]

        # txt
        t_txt = self.encoder_vl(txt, mask, mode='txt', use_mask=use_mask, percent=percent)  # [4, 128, 768]
        out_txt = t_txt[:, 0]                           # [4, 768]

        if self.training:
            return out_vis, out_txt
        else:
            return out_vis, out_txt

    def forward_hybrid(self, img, vis, txt, mask):  # [24, 3, 256, 128]
        # img
        t_img = self.encoder_vl(img, mode='img')        # [4, 128, 768]
        img_feat = t_img[:, 0]                          # [4, 768]
        img_feat_bn = self.bottleneck_img(img_feat)     # [4, 2048]

        # vis
        t_vis = self.encoder_vl(vis, mode='vis')        # [4, 128, 768]
        out_vis = t_vis[:, 0]                           # [4, 768]

        # txt
        t_txt = self.encoder_vl(txt, mask, mode='txt')  # [4, 128, 768]
        out_txt = t_txt[:, 0]                           # [4, 768]

        return img_feat_bn, out_vis, out_txt

    def forward_token(self, vis, txt, mask, use_mask=False, percent=0.2):  # [24, 3, 256, 128]
        # vis
        t_vis = self.encoder_vl(vis, mode='vis', use_mask=use_mask, percent=percent)        # [4, 128, 768]
        t_vis = F.normalize(t_vis, dim=-1)[:, 1:]

        # txt
        t_txt = self.encoder_vl(txt, mask, mode='txt', use_mask=use_mask, percent=percent)  # [4, 128, 768]
        t_txt = F.normalize(t_txt, dim=-1)[:, 0:1]

        return t_vis, t_txt


    def forward_txt2img(self, vis, txt, mask):  # [24, 3, 256, 128]
        # vis
        t_vis = self.encoder_vl(vis, mode='vis')        # [4, 128, 768]
        out_vis = t_vis[:, 0]                           # [4, 768]

        # txt
        t_txt = self.encoder_vl(txt, mask, mode='txt')  # [4, 128, 768]
        out_txt = t_txt[:, 0]                           # [4, 768]

        return out_vis, out_txt

    def forward_test(self, x, mask=None, mode='img'):           # [24, 3, 256, 128]
        # encoding
        t = self.encoder_vl(x, mask, mode)                      # [4, 128, 768]
        t = t[:, 0]

        # head projection
        if mode == 'txt':
            out_list = F.normalize(t, dim=-1)
        elif mode == 'vis':
            out_list = F.normalize(t, dim=-1)                   # [4, 2048]
        else:
            img_feat_bn = self.bottleneck_img(t)                # [4, 768]
            out_list = img_feat_bn

        return out_list


    def forward_img(self, img):  # [24, 3, 256, 128]
        # img
        t_img = self.encoder_vl(img, mode='img')        # [4, 128, 768]
        img_feat = t_img[:, 0]                          # [4, 768]
        img_feat_bn = self.bottleneck_img(img_feat)     # [4, 768]

        return img_feat_bn

    def forward_txt(self, txt, mask):  # [24, 3, 256, 128]
        # txt
        t_txt = self.encoder_vl(txt, mask, mode='txt')  # [4, 128, 768]
        out_txt = t_txt[:, 0]                           # [4, 768]

        return out_txt


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))



class build_transformer_uvt_img_txt_pretrain(nn.Module):
    def __init__(self, num_classes, cfg, factory, context_length=64):
        super(build_transformer_uvt_img_txt_pretrain, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        bert_path = cfg.MODEL.BERT_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.feat_dim = 768
        self.use_mask = cfg.MODEL.USE_MASK
        # temp 默认0.07
        self.temp = nn.Parameter(torch.ones([]) * cfg.MODEL.TEMP)
        self.context_length = context_length

        ## base
        bert_config = BertConfig.from_json_file(cfg.TEXT.BERT_CONFIG)
        text_encoder = BertModel.from_pretrained(cfg.TEXT.TEXT_ENCODER, config=bert_config, add_pooling_layer=False)

        self.encoder_vl = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                              drop_rate=cfg.MODEL.DROP_OUT, attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, config=bert_config)

        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.encoder_vl.load_param_hybrid(model_path, text_encoder)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.encoder_fusion = BertForMaskedLM_F.from_pretrained(cfg.TEXT.TEXT_ENCODER, config=bert_config)

        self.temp = nn.Parameter(torch.ones([]) * cfg.MODEL.TEMP)  # temp 默认0.07

        ## use_matching_loss:
        self.itm = nn.Linear(self.feat_dim, 2)
        # self.itm = build_mlp(input_dim=self.feat_dim, output_dim=2)  # 768->768*2->768

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes  # 11003
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    def encode_text(self, text):  # [24, 77]
        x = self.token_embedding(text).type(self.dtype)  # [24, 77, 512]
        x = x + self.positional_embedding.type(self.dtype).expand_as(x)  # [24, 77, 512]

        x = x.permute(1, 0, 2)  # [77, 24, 512]
        x = self.text_encoder(x)  # [77, 24, 512]
        x = x.permute(1, 0, 2)  # [24, 77, 512]
        x = self.ln_final(x).type(self.dtype)  # [24, 77, 512]

        return x  # [24, 77, 512]

    def embed_position(self, x, mode='img'):        # [4, 64]
        if mode == 'txt':
            x = self.token_embedding(x).type(self.dtype)    # [4, 64, 768]
            x = x + self.pos_embed_txt.type(self.dtype).expand_as(x)  # [24, 64, 768]
        else:
            x = self.patch_embed(x)     # [64, 128, 768]
            x = x + self.pos_embed
        return x

    def forward(self, vis, txt, mask, txt_masked=None, masked_pos=None, masked_ids=None, pid=None):  # [24, 3, 256, 128]
        # vis
        t_vis = self.encoder_vl(vis, mode='vis')        # [4, 128, 768]
        out_vis = t_vis[:, 0]                           # [4, 768]

        # txt
        t_txt = self.encoder_vl(txt, mask, mode='txt')  # [4, 128, 768]
        out_txt = t_txt[:, 0]                           # [4, 768]

        if self.training:
            img_atts = torch.ones(t_vis.size()[:-1], dtype=torch.long).to(t_vis.device)
            loss_itc = self.get_contrastive_loss(out_vis, out_txt, pid)             # 28.4028
            loss_itm = self.get_matching_loss(t_vis, img_atts, out_vis, t_txt, mask, out_txt)  # 1.5443
            loss_mlm = self.get_mlm_loss(txt_masked, mask, t_vis, img_atts, masked_pos, masked_ids)
            return loss_itc, loss_itm, loss_mlm
        else:
            return out_vis, out_txt

    def forward_txt2img(self, vis, txt, mask):  # [24, 3, 256, 128]
        # vis
        t_vis = self.encoder_vl(vis, mode='vis')        # [4, 128, 768]
        out_vis = t_vis[:, 0]                           # [4, 768]

        # txt
        t_txt = self.encoder_vl(txt, mask, mode='txt')  # [4, 128, 768]
        out_txt = t_txt[:, 0]                           # [4, 768]

        return out_vis, out_txt


    def forward_img(self, img):  # [24, 3, 256, 128]
        # img
        t_img = self.encoder_vl(img, mode='img')        # [4, 128, 768]
        img_feat = t_img[:, 0]                          # [4, 768]
        img_feat_bn = self.bottleneck_img(img_feat)     # [4, 768]

        return img_feat_bn

    def forward_txt(self, txt, mask):  # [24, 3, 256, 128]
        # txt
        t_txt = self.encoder_vl(txt, mask, mode='txt')  # [4, 128, 768]
        out_txt = t_txt[:, 0]                           # [4, 768]

        return out_txt


    def get_cross_embeds(self, image_embeds, image_atts, text_embeds=None, text_atts=None):
        assert text_atts is not None

        encoder = self.encoder_fusion.bert

        return encoder(encoder_embeds=text_embeds,  # [8, 13, 768]
                       attention_mask=text_atts,
                       encoder_hidden_states=image_embeds,  # [8, 193, 768]
                       encoder_attention_mask=image_atts,
                       return_dict=True,
                       mode='fusion',
                       ).last_hidden_state


    # # distribute training
    # def get_contrastive_loss(self, image_feat, text_feat, idx=None):
    #     """
    #     Args:
    #         image_feat, text_feat: normalized
    #
    #     Returns: contrastive loss
    #
    #     """
    #     assert image_feat.size(-1) == self.embed_dim
    #     assert text_feat.size(-1) == self.embed_dim
    #
    #     image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
    #     text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
    #     logits = image_feat_all @ text_feat_all.t() / self.temp
    #
    #     bsz = image_feat_all.shape[0]
    #
    #     if idx is None:
    #         labels = torch.arange(bsz, device=image_feat.device)
    #         loss_i2t = F.cross_entropy(logits, labels)
    #         loss_t2i = F.cross_entropy(logits.t(), labels)
    #
    #     else:
    #         idx = idx.view(-1, 1)
    #         assert idx.size(0) == image_feat.size(0)
    #         idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
    #         pos_idx = torch.eq(idx_all, idx_all.t()).float()
    #         labels = pos_idx / pos_idx.sum(1, keepdim=True)
    #
    #         loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
    #         loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()
    #
    #     return (loss_i2t + loss_t2i) / 2

    # sing gpu training
    def get_contrastive_loss(self, image_feat, text_feat, idx=None):   # [4, 768], [4, 768]
        """
        Args:
            image_feat, text_feat: normalized

        Returns: contrastive loss

        """
        assert image_feat.size(-1) == self.feat_dim
        assert text_feat.size(-1) == self.feat_dim

        image_feat_all = image_feat  # [8, 256]
        text_feat_all = text_feat  # [8, 256]
        logits = image_feat_all @ text_feat_all.t() / self.temp  # [8, 8], temp=0.5

        bsz = image_feat_all.shape[0]

        if idx is None:
            labels = torch.arange(bsz, device=image_feat.device)  # [0, 1, 2, 3, 4, 5, 6, 7]
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)

        else:
            idx = idx.view(-1, 1)
            assert idx.size(0) == image_feat.size(0)
            idx_all = idx
            pos_idx = torch.eq(idx_all, idx_all.t()).float()
            labels = pos_idx / pos_idx.sum(1, keepdim=True)

            loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()

        return (loss_i2t + loss_t2i) / 2

    def get_matching_loss(self, image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx=None):
        """
        Matching Loss with hard negatives
        """
        bs = image_embeds.size(0)
        with torch.no_grad():
            sim_i2t = image_feat @ text_feat.t() / self.temp            # [8, 8]
            sim_t2i = text_feat @ image_feat.t() / self.temp            # [8, 8]

            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4

            if idx is None:
                weights_i2t.fill_diagonal_(0)
                weights_t2i.fill_diagonal_(0)
            else:
                idx = idx.view(-1, 1)
                assert idx.size(0) == bs
                mask = torch.eq(idx, idx.t())
                weights_i2t.masked_fill_(mask, 0)
                weights_t2i.masked_fill_(mask, 0)

        image_embeds_neg = []
        image_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
            image_atts_neg.append(image_atts[neg_idx])

        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        image_atts_neg = torch.stack(image_atts_neg, dim=0)

        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])

        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts_neg, image_atts], dim=0)

        cross_pos = self.get_cross_embeds(image_embeds, image_atts, text_embeds=text_embeds, text_atts=text_atts)[:, 0, :]      # [8, 768]
        cross_neg = self.get_cross_embeds(image_embeds_all, image_atts_all, text_embeds=text_embeds_all, text_atts=text_atts_all)[:, 0, :]  # [16, 768]

        output = self.itm(torch.cat([cross_pos, cross_neg], dim=0))     # [24, 2]
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(image_embeds.device)

        return F.cross_entropy(output, itm_labels)


    # def get_mlm_loss(self, text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids):
    #     text_mask_emb = self.encoder_text.bert(text_ids_masked, text_atts).last_hidden_state
    #     return self.encoder_fusion_t2i(encoder_embeds=text_mask_emb,
    #                                    attention_mask=text_atts,
    #                                    encoder_hidden_states=image_embeds,
    #                                    encoder_attention_mask=image_atts,
    #                                    return_dict=True,
    #                                    mode='fusion',
    #                                    labels=masked_ids,
    #                                    masked_pos=masked_pos).loss


    def get_mlm_loss(self, txt_masked, mask, t_vis, img_atts, masked_pos, masked_ids):
        text_mask_emb = self.encoder_vl(txt_masked, mask, mode='txt')
        return self.encoder_fusion(encoder_embeds=text_mask_emb,
                                       attention_mask=mask,
                                       encoder_hidden_states=t_vis,
                                       encoder_attention_mask=img_atts,
                                       return_dict=True,
                                       mode='fusion',
                                       labels=masked_ids,
                                       masked_pos=masked_pos,
                                       ).loss

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()  # [8, 13]

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))







__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID,
    'vit_base_patch16_224_uvt_img_txt': vit_base_patch16_224_uvt_img_txt,
    'vit_base_patch16_224_uvt_img_txt_mask': vit_base_patch16_224_uvt_img_txt_mask,
}


def make_model(cfg, num_class, camera_num=0, view_num=0, tokenizer=None, num_fusion=2):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)  # this
            print('===========building transformer===========')
    elif cfg.MODEL.NAME == 'transformer_uvt_img_txt_mask':
        model = build_transformer_uvt_img_txt_mask(num_class, cfg, __factory_T_type)
    elif cfg.MODEL.NAME == 'transformer_uvt_img_txt_pretrain':
        model = build_transformer_uvt_img_txt_pretrain(num_class, cfg, __factory_T_type)
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model




