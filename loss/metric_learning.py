import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.nn import Parameter
import math
import numpy as np

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.3, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs, inputs.t())
        targets = targets
        loss = list()
        c = 0

        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets == targets[i])

            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)
            neg_pair_ = torch.masked_select(sim_mat[i], targets != targets[i])

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]

            neg_pair = torch.masked_select(neg_pair_, neg_pair_ > self.margin)

            neg_loss = 0

            pos_loss = torch.sum(-pos_pair_ + 1)
            if len(neg_pair) > 0:
                neg_loss = torch.sum(neg_pair)
            loss.append(pos_loss + neg_loss)

        loss = sum(loss) / n
        return loss


class CircleLoss(nn.Module):
    def __init__(self, in_features, num_classes, s=256, m=0.25):
        super(CircleLoss, self).__init__()
        self.weight = Parameter(torch.Tensor(num_classes, in_features))     # [128, 512]
        self.s = s
        self.m = m
        self._num_classes = num_classes     # 128
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def __call__(self, bn_feat, targets):       # [16, 512], [16,]

        sim_mat = F.linear(F.normalize(bn_feat), F.normalize(self.weight))      # [16, 128], min=-0.1299, max=0.1654
        alpha_p = torch.clamp_min(-sim_mat.detach() + 1 + self.m, min=0.)       # [16, 128]
        alpha_n = torch.clamp_min(sim_mat.detach() + self.m, min=0.)            # [16, 128]
        delta_p = 1 - self.m        # 0.75
        delta_n = self.m            # 0.25

        s_p = self.s * alpha_p * (sim_mat - delta_p)         # [16, 128]
        s_n = self.s * alpha_n * (sim_mat - delta_n)         # [16, 128]

        targets = F.one_hot(targets, num_classes=self._num_classes)     # [16, 128]

        pred_class_logits = targets * s_p + (1.0 - targets) * s_n       # [16, 128]

        return pred_class_logits


class Arcface(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.30, easy_margin=False, ls_eps=0.0):
        super(Arcface, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s              # 30
        self.m = m              # 0.3
        self.ls_eps = ls_eps    # 0.0, label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))       # [512, 512]
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin          # False
        self.cos_m = math.cos(m)                # 0.955
        self.sin_m = math.sin(m)                # 0.295
        self.th = math.cos(math.pi - m)         # -0.955
        self.mm = math.sin(math.pi - m) * m     # 0.088

    def forward(self, input, label):        # [16, 512], [16,]
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))         # [16, 512]
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))       # [16, 512]
        phi = cosine * self.cos_m - sine * self.sin_m       # [16, 512]
        phi = phi.type_as(cosine)           # [16, 512]
        if self.easy_margin:                # False
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)          # [16, 512]
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)            # [16, 512]
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)       # [16, 512]
        output *= self.s

        return output


class Cosface(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.30):
        super(Cosface, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class AMSoftmax(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.30):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_features
        self.W = torch.nn.Parameter(torch.randn(in_features, out_features), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        # print(x_norm.shape, w_norm.shape, costh.shape)
        lb_view = lb.view(-1, 1)
        delt_costh = torch.zeros(costh.size(), device='cuda').scatter_(1, lb_view, self.m)
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        return costh_m_s




if __name__ == '__main__':
    inputs = F.normalize(torch.randn((16, 512)), dim=-1).cuda()
    targets = torch.from_numpy(np.array([[item] * 4 for item in range(int(inputs.shape[0] / 4))]).reshape(-1)).cuda()

    # criterion = ContrastiveLoss().cuda()
    # criterion = CircleLoss(in_features=inputs.shape[-1], num_classes=128).cuda()
    criterion = Arcface(in_features=inputs.shape[-1], out_features=inputs.shape[-1]).cuda()
    # criterion = Cosface(in_features=inputs.shape[-1], out_features=inputs.shape[-1]).cuda()

    loss = criterion(inputs, targets)
    print('loss = ', loss.item())

    print('finish')






