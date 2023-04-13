################################################################################
#
# Implementation of angular additive margin softmax loss.
#
# Adapted from: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
#
# Author(s): Nik Vaessen
################################################################################

import torch

import torch as t
import torch.nn as nn
import torch.nn.functional as F

import math


################################################################################
# wrap around aam-loss implementation


class AngularAdditiveMarginSoftMaxLoss(t.nn.Module):
    def __init__(
            self,
            input_features,
            output_features,
            margin=0.3,
            scale=15,
            easy_margin=False,
    ):
        super(AngularAdditiveMarginSoftMaxLoss, self).__init__()

        self.margin = margin
        self.scale = scale
        self.input_features = input_features
        self.fc_weights = torch.nn.Parameter(
            torch.FloatTensor(output_features, input_features), requires_grad=True
        )
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.fc_weights, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, x, label=None):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.input_features

        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.fc_weights))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        # one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale

        loss = self.ce(output, label)
        prediction = F.softmax(output, dim=1)

        return loss, prediction


if __name__ == '__main__':
    # recreate AAM loss with correct input/output features
    loss_fn = AngularAdditiveMarginSoftMaxLoss(
        input_features=768,
        output_features=4,
    )

    x = torch.randn(4, 768)
    y = torch.tensor([0, 1, 2, 3])
    loss, pred_ = loss_fn(x, y)
    print(
        loss,pred_
    )
    # tensor(6.0972, grad_fn=<NllLossBackward0>) torch.Size([4, 4])