import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, num_classes=5000, feature_size=768, resume=False, epsilon=1e-8):
        super(Loss, self).__init__()
        self.CMPM = True
        self.epsilon = epsilon
        self.num_classes = num_classes
        if resume:
            checkpoint = torch.load(resume)
            self.W = Parameter(checkpoint['W'])
            print('=========> Loading in parameter W from pretrained models')
        else:
            self.W = Parameter(torch.randn(feature_size, num_classes))  # [2048, 11003]
            self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.W.data, gain=1)

    def forward(self, image_embeddings, text_embeddings, labels):  # [8, 768], [8, 768], [8,]
        """
        Cross-Modal Projection Matching Loss(CMPM)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
            i2t_loss: cmpm loss for image projected to text
            t2i_loss: cmpm loss for text projected to image
            pos_avg_sim: average cosine-similarity for positive pairs
            neg_avg_sim: averate cosine-similarity for negative pairs
        """

        batch_size = image_embeddings.shape[0]  # 8
        labels_reshape = torch.reshape(labels, (batch_size, 1))  # [8, 1]
        labels_dist = labels_reshape - labels_reshape.t()  # [8, 8]
        labels_mask = (labels_dist == 0)  # [8, 8]

        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)  # [8, 768]
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)  # [8, 768]
        image_proj_text = torch.matmul(image_embeddings, text_norm.t())  # [8, 8]
        text_proj_image = torch.matmul(text_embeddings, image_norm.t())  # [8, 8]

        # normalize the true matching distribution
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)  # [8, 8]

        i2t_pred = F.softmax(image_proj_text, dim=1)  # [8, 8]
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + self.epsilon))  # [8, 8]
        t2i_pred = F.softmax(text_proj_image, dim=1)  # [8, 8]
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + self.epsilon))  # [8, 8]

        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))  # 15.0354

        return cmpm_loss  # 28.2706



# class LossVT(nn.Module):
#     def __init__(self, num_classes=5000, feature_size=768, resume=False, epsilon=1e-8):
#         super(LossVT, self).__init__()
#         self.CMPM = True
#         self.epsilon = epsilon
#         self.num_classes = num_classes
#         # if resume:
#         #     checkpoint = torch.load(resume)
#         #     self.W = Parameter(checkpoint['W'])
#         #     print('=========> Loading in parameter W from pretrained models')
#         # else:
#         #     self.W = Parameter(torch.randn(feature_size, num_classes))  # [2048, 11003]
#         #     self.init_weight()
#
#     # def init_weight(self):
#     #     nn.init.xavier_uniform_(self.W.data, gain=1)
#
#     def forward(self, image_embeddings, text_embeddings, labels_img, labels_txt):  # [8, 768], [8, 768], [8,]
#         """
#         Cross-Modal Projection Matching Loss(CMPM)
#         :param image_embeddings: Tensor with dtype torch.float32
#         :param text_embeddings: Tensor with dtype torch.float32
#         :param labels: Tensor with dtype torch.int32
#         :return:
#             i2t_loss: cmpm loss for image projected to text
#             t2i_loss: cmpm loss for text projected to image
#             pos_avg_sim: average cosine-similarity for positive pairs
#             neg_avg_sim: averate cosine-similarity for negative pairs
#         """
#
#         batch_size = image_embeddings.shape[0]  # 8
#         labels_reshape_img = torch.reshape(labels_img, (batch_size, 1))  # [8, 1]
#         labels_reshape_txt = torch.reshape(labels_txt, (batch_size, 1))  # [8, 1]
#         labels_dist_img = labels_reshape_img - labels_reshape_txt.t()  # [8, 8]
#         labels_mask_img = (labels_dist_img == 0)  # [8, 8]
#         labels_dist_txt = labels_reshape_txt - labels_reshape_img.t()  # [8, 8]
#         labels_mask_txt = (labels_dist_txt == 0)  # [8, 8]
#
#         image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)  # [8, 768]
#         text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)  # [8, 768]
#         image_proj_text = torch.matmul(image_embeddings, text_norm.t())  # [8, 8]
#         text_proj_image = torch.matmul(text_embeddings, image_norm.t())  # [8, 8]
#
#         # normalize the true matching distribution
#         labels_mask_norm_img = labels_mask_img.float() / (labels_mask_img.float().norm(dim=1) + self.epsilon)  # [8, 8]
#         i2t_pred = F.softmax(image_proj_text, dim=1)  # [8, 8]
#         i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm_img + self.epsilon))  # [8, 8]
#
#         labels_mask_norm_txt = labels_mask_txt.float() / (labels_mask_txt.float().norm(dim=1) + self.epsilon)  # [8, 8]
#         t2i_pred = F.softmax(text_proj_image, dim=1)  # [8, 8]
#         t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm_txt + self.epsilon))  # [8, 8]
#
#         cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))  # 15.0354
#
#         return cmpm_loss  # 28.2706

