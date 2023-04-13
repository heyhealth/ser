import sys
import os

sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))
from torch import nn
import torch
from models.wav2vec2_conv_modules1 import ConvFeatureExtractionModel
from models.wav2vec2_wrappers import Wav2vec2Wrapper
from main.opts import ARGS
from models.DownStreamModels import MLP
from models.wav2vec2_components import get_feat_extract_output_lengths

"""
The Vanilla Fine-tuning
"""


class Wav2Vec2EmoNet(nn.Module):

    def __init__(self, num_classes):
        super(Wav2Vec2EmoNet, self).__init__()
        # use the wav2vec2 as the backend
        self.backend = 'wav2vec2'
        feature_dim = 768
        self.wav2vec2 = Wav2vec2Wrapper(pretrain=False)
        self.rnn_head = nn.LSTM(feature_dim, 256, 1, bidirectional=True)
        self.linear_head = nn.Sequential(
            # maybe the relu is redundant
            nn.ReLU(),
            nn.Linear(768, num_classes)
        )

    def trainable_parameters(self):
        return list(self.rnn_head.parameters()) + list(self.linear_head.parameters()) + list(
            getattr(self, self.backend).trainable_params())

    def forward(self, x, length):
        reps = getattr(self, self.backend)(x, length)
        last_feat_pos = get_feat_extract_output_lengths(length) - 1
        logits = reps.permute(1, 0, 2)  # L, B, C
        masks = torch.arange(logits.size(0), device=logits.device).expand(last_feat_pos.size(0),
                                                                          -1) < last_feat_pos.unsqueeze(1)
        masks = masks.float()
        logits = (logits * masks.T.unsqueeze(-1)).sum(0) / last_feat_pos.unsqueeze(1)
        logits = self.linear_head(logits)
        return logits





if __name__ == '__main__':
    net = Wav2Vec2EmoNet(num_classes=4)

    x = torch.randn((4, 128000), device='cpu')
    length = torch.tensor([128000, 12800, 12800, 12800], dtype=torch.long, device='cpu')

    print(net(x, length).shape)
