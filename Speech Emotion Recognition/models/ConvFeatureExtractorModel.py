import os
import sys

sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))
import torch
from torch import nn
from torch.nn import functional as F
from models.wav2vec2_conv_modules1 import Wav2Vec2FeatureEncoderWrapper
from models.DownStreamModels import MLP, RNNLayer
from models.wav2vec2_components import get_feat_extract_output_lengths

"""
Only use the wav2vec2 ConvFeatureExtractor module
"""


class Wav2Vec2EmoNet(nn.Module):

    def __init__(self, num_classes):
        super(Wav2Vec2EmoNet, self).__init__()

        self.wav2vec2conv = Wav2Vec2FeatureEncoderWrapper()
        self.classifier = nn.Linear(768, num_classes)
        # self.rnn = RNNLayer()
        self.reset_parameters()

    def trainable_parameters(self):
        return list(self.wav2vec2conv.parameters()) + list(self.classifier.parameters())

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x, length):
        reps = self.wav2vec2conv(x)
        last_feat_pos = get_feat_extract_output_lengths(length)
        logits = reps.permute(1, 0, 2)  # L, B, C
        masks = torch.arange(logits.size(0), device=logits.device).expand(last_feat_pos.size(0),
                                                                          -1) < last_feat_pos.unsqueeze(1)
        masks = masks.float()
        logits = (logits * masks.T.unsqueeze(-1)).sum(0) / last_feat_pos.unsqueeze(1)
        logits = self.classifier(logits)
        return logits


if __name__ == '__main__':
    x = torch.randn((1, 128000))
    net = Wav2Vec2EmoNet(num_classes=4)
    print(
        net(x, torch.tensor([12500])).shape
    )
