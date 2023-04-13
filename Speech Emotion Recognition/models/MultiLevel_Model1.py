"""
inspired from SPEECH EMOTION RECOGNITION WITH CO-ATTENTION BASED MULTI-LEVEL ACOUSTIC INFORMATION
"""

import collections

import numpy as np
from torch import nn
import torch
from models.wav2vec2_conv_modules4 import Wav2Vec2EncoderWrapper
from models.wav2vec2_components import get_pretrained_wav2vec2_model, global_avg_pooling1d_with_masks, \
    get_feat_extract_output_lengths
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Config, Wav2Vec2FeatureProjection
from models.wav2vec2_conv_modules3 import Wav2Vec2FeatureEncoder
from mydatasets.feature_acoustic import extract_mfcc_from_wavefrom, extract_melspectrogram_from_waveform
from models.DeepLearningModels import SER_AlexNet

class Wav2Vec2EmoNet(nn.Module):

    def __init__(self, num_classes):
        super(Wav2Vec2EmoNet, self).__init__()
        # Wav2Vec2 config
        self.config = Wav2Vec2Config(
            gradient_checkpointing=False,
            # SpecAugment
            apply_spec_augment=True,
            mask_time_length=10,
            mask_time_prob=0.08,
            mask_feature_length=64,
            mask_feature_prob=0.05,
            mask_time_min_masks=2,
        )

        self.mfcc_encoder = nn.Sequential()

        self.mel_spec_encoder = SER_AlexNet(num_classes,in_ch=3,pretrained=True)

        # Wav2Vec2 feature_extractor
        self.feature_extractor = Wav2Vec2FeatureEncoder(self.config)

        # Wav2Vec2 feature_projection
        self.feature_projection = Wav2Vec2FeatureProjection(self.config)

        # Wav2Vec2 transformer encoder
        self.encoder = Wav2Vec2EncoderWrapper(config=self.config)

        # classifier head
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(768, num_classes)
        )
        # initialize model parameters
        self.reset_parameters()

    def _compute_MFCC_feature(self, x):
        """
        :param x:[B,waveform]
        :return: MFCC:[B,20,L]
        """
        with torch.no_grad():
            waveform = x[0]
            mfcc = extract_mfcc_from_wavefrom(waveform.numpy(), sr=16000, n_fft=2048,
                                              hop_length=512, fmax=8000)
            mfcc_ = torch.as_tensor(mfcc, dtype=x.dtype, device=x.device).unsqueeze(0)
            for i in range(x.shape[0] - 1):
                mfcc = extract_mfcc_from_wavefrom(waveform.numpy(), sr=16000, n_fft=2048,
                                                  hop_length=512, fmax=8000)
                mfcc_ = np.concatenate([mfcc_, torch.as_tensor(mfcc, dtype=x.dtype, device=x.device).unsqueeze(0)],
                                       axis=0)
        return mfcc_

    def _compute_Mel_Spectrogram_feature(self, x):
        """
        :param x: [B,waveform]
        :return: Mel_Spectrogram:[B,128,L]
        """
        with torch.no_grad():
            waveform = x[0]
            mel_spec = extract_melspectrogram_from_waveform(waveform.numpy(), sr=16000, n_fft=2048,
                                                            hop_length=512, fmax=8000)
            mel_spec_ = torch.as_tensor(mel_spec, dtype=x.dtype, device=x.device).unsqueeze(0)
            for i in range(x.shape[0] - 1):
                mel_spec = extract_melspectrogram_from_waveform(waveform.numpy(), sr=16000, n_fft=2048,
                                                                hop_length=512, fmax=8000)
                mel_spec_ = np.concatenate(
                    [mel_spec_, torch.as_tensor(mel_spec, dtype=x.dtype, device=x.device).unsqueeze(0)],
                    axis=0)
        return mel_spec_

    def forward(self, x, length):

        with torch.no_grad():
            # MFCC  [B,C(20),L]
            mfcc_feature = self._compute_MFCC_feature(x)
            # Mel Spectrogram  [B,C(128),L]
            mel_spectrogram_feature = self._compute_Mel_Spectrogram_feature(x)
            x, all_layers_hidden_states = self.feature_extractor(x)
            hierarchical_feature = torch.zeros(x.shape[0], self.config.conv_dim[-1], dtype=torch.float, device=x.device)
            for layer_index, layer_hidden_states in enumerate(all_layers_hidden_states):
                layer_hidden_states = layer_hidden_states.transpose(1, 2)
                hierarchical_feature += global_avg_pooling1d_with_masks(layer_hidden_states, length,
                                                                        conv_kernel=self.config.conv_kernel[
                                                                                    :layer_index + 1],
                                                                        conv_stride=self.config.conv_stride[
                                                                                    :layer_index + 1])

            hierarchical_feature = hierarchical_feature.unsqueeze(-1)
            # try different fusion style
            x = torch.cat([x, hierarchical_feature], dim=-1)
            x = x.transpose(1, 2)

        hidden_states, _ = self.feature_projection(x)

        # ----- mfcc encoder -----
        mfcc_hidden_states = self.mfcc_encoder(mfcc_feature)
        # ----- mel spec encoder -----
        mel_spec_hidden_states = self.mel_spec_encoder(mel_spectrogram_feature)

        # ----- wav2vec2 encoder -----
        hidden_states = self.encoder(hidden_states, length)
        # ----- global avg pooling with masks -----
        embedding = global_avg_pooling1d_with_masks(hidden_states, length, self.config.conv_kernel,
                                                    self.config.conv_stride)
        """
        last_feat_pos = get_feat_extract_output_lengths(length, conv_kernel=self.config.conv_kernel,
                                                        conv_stride=self.config.conv_stride) - 1
        logits = hidden_states.permute(1, 0, 2)  # L, B, C
        masks = torch.arange(logits.size(0), device=logits.device).expand(last_feat_pos.size(0),
                                                                          -1) < last_feat_pos.unsqueeze(1)
        masks = masks.float()
        logits = (logits * masks.T.unsqueeze(-1)).sum(0) / last_feat_pos.unsqueeze(1)
        """
        # ----- classifier head -----
        logits = self.classifier(embedding)
        return logits

    def reset_parameters(self):
        """initialize the modules' parameters"""
        pretrained_wav2vec2_model = get_pretrained_wav2vec2_model()
        pretrained_feature_extractor_state_dict = pretrained_wav2vec2_model.feature_extractor.state_dict()
        pretrained_feature_projection_state_dict = pretrained_wav2vec2_model.feature_projection.state_dict()
        self.feature_extractor.load_state_dict(pretrained_feature_extractor_state_dict)
        self.feature_projection.load_state_dict(pretrained_feature_projection_state_dict)

    def trainable_parameters(self):
        params = list(self.feature_projection.parameters()) + list(self.encoder.trainable_parameters()) + list(
            self.classifier.parameters())
        return params


if __name__ == '__main__':
    """
    # wav2vec2 = Wav2Vec2Model.from_pretrained(ARGS.MODEL_NAME_Wav2Vec2)
    # print(
    #     wav2vec2
    # )
    """
    x = torch.randn(4, 160000)
    length = torch.tensor([128000, 12300, 12300, 1234], dtype=torch.long)
    net = Wav2Vec2EmoNet(num_classes=12)
    print(
        net(x, length)
    )
    """
    x = torch.randn(4, 399, 768)
    print(
        global_avg_pooling1d_with_masks(x, length).shape
    )
    """
    # torch.Size([4, 160000])
    # tensor([35200, 160000, 115571, 27560])
