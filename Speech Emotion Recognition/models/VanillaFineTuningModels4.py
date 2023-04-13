import collections
import os

from torch import nn
import torch
from models.wav2vec2_conv_modules4 import Wav2Vec2EncoderWrapper
from models.wav2vec2_components import get_feat_extract_output_lengths, get_pretrained_wav2vec2_model, \
    get_pretrained_wav2vec2_model_config
from models.wavlm_components import get_pretrained_wavlm_model_config, get_pretrained_wavlm_model
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Config, Wav2Vec2FeatureProjection
from transformers.models.wavlm.modeling_wavlm import WavLMFeatureProjection, WavLMFeatureEncoder, WavLMConfig
from models.wav2vec2_conv_modules2 import Wav2Vec2FeatureEncoder
from models.wavlm_wrappers import WavLMWrapper, WavLMEncoderWrapper
from main.opts import ARGS
from utils.pooling_v3 import IndexPool1D

"""
first & cls
"""


class Wav2Vec2EmoNet(nn.Module):

    def __init__(self, num_classes):
        super(Wav2Vec2EmoNet, self).__init__()
        # Wav2Vec2 config
        self.config = Wav2Vec2Config(
            gradient_checkpointing=False,
            # SpecAugment
            apply_spec_augment=True,
            mask_time_length=ARGS.AUDIO_SEGMENT_TIME_LENGTH,
            mask_time_prob=0.08,
            mask_feature_length=64,
            mask_feature_prob=0.05,
            mask_time_min_masks=2,
            use_weighted_layer_sum=True,
        )
        # add cls token
        self.cls_token_constant = 1

        # Wav2Vec2 feature_extractor
        self.feature_extractor = Wav2Vec2FeatureEncoder(self.config)

        # Wav2Vec2 feature_projection
        self.feature_projection = Wav2Vec2FeatureProjection(self.config)

        # Wav2Vec2 transformer encoder
        # TODO:优化加载参数时间
        self.encoder = Wav2Vec2EncoderWrapper(self.config)

        self.pooling = IndexPool1D(
            selection_method="first+cls", dim_to_reduce=1
        )

        # classifier head
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(768, num_classes)
        )
        # initialize model parameters
        self.reset_parameters()

    def forward(self, x, length):
        with torch.no_grad():
            x = self.feature_extractor(x)
            x = x.transpose(1, 2)
            hidden_states, _ = self.feature_projection(x)

        # add cls token
        cls_token = (
                torch.ones((x.shape[0], 1, 768), device=x.device)
                * self.cls_token_constant
        )
        hidden_states = torch.cat([cls_token, hidden_states], dim=1)

        # ----- wav2vec2 encoder -----
        hidden_states = self.encoder(hidden_states, length)
        # pooling
        embedding = self.pooling(hidden_states)
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
        params = list(self.encoder.trainable_parameters()) + list(
            self.classifier.parameters())
        return params


# first & cls
class WavLMEmoNet(nn.Module):

    def __init__(self, num_classes):
        super(WavLMEmoNet, self).__init__()
        # pretrained wavlm
        self.config = WavLMConfig(
            gradient_checkpointing=False,
            # SpecAugment
            apply_spec_augment=True,
            mask_time_length=ARGS.AUDIO_SEGMENT_TIME_LENGTH,
            mask_time_prob=0.08,
            mask_feature_length=64,
            mask_feature_prob=0.05,
            mask_time_min_masks=2,
        )

        # add cls token
        self.cls_token_constant = 1

        # need WavLM-> WavLM ConvFeatureEncoder + WavLM Encoder
        # self.wavlm = WavLMWrapper(config=self.config)

        # conv feature encoder
        self.feature_extractor = WavLMFeatureEncoder(self.config)
        # feature projection
        self.feature_projection = WavLMFeatureProjection(self.config)
        # transformer encoder
        self.encoder = WavLMEncoderWrapper(self.config)
        # pooling method
        self.pooling = IndexPool1D(
            selection_method="first+cls", dim_to_reduce=1
        )

        # classifier head
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(768, num_classes)
        )
        # load the pretrained parameters
        self.reset_parameters()

    def forward(self, x, length):
        with torch.no_grad():
            x = self.feature_extractor(x)
            x = x.transpose(1, 2)
            hidden_states, _ = self.feature_projection(x)

        # add cls token
        cls_token = (
                torch.ones((x.shape[0], 1, 768), device=x.device)
                * self.cls_token_constant
        )
        hidden_states = torch.cat([cls_token, hidden_states], dim=1)

        # ----- wavlm encoder -----
        hidden_states = self.encoder(hidden_states, length)
        # pooling
        embedding = self.pooling(hidden_states)

        # ----- classifier head -----
        logits = self.classifier(embedding)
        return logits

    def trainable_parameters(self):
        return list(self.encoder.trainable_parameters()) + list(self.classifier.parameters())

    def reset_parameters(self):
        """initialize the modules' parameters"""
        pretrained_wavlm_model = get_pretrained_wavlm_model()
        pretrained_feature_extractor_state_dict = pretrained_wavlm_model.feature_extractor.state_dict()
        pretrained_feature_projection_state_dict = pretrained_wavlm_model.feature_projection.state_dict()
        self.feature_extractor.load_state_dict(pretrained_feature_extractor_state_dict)
        self.feature_projection.load_state_dict(pretrained_feature_projection_state_dict)


if __name__ == '__main__':
    """
    # wav2vec2 = Wav2Vec2Model.from_pretrained(ARGS.MODEL_NAME_Wav2Vec2)
    # print(
    #     wav2vec2
    # )
    """
    # x = torch.randn(4, 128000)
    # length = torch.tensor([128000, 12300, 12300, 1234], dtype=torch.long)
    net = Wav2Vec2EmoNet(num_classes=5)
    # print(
    #     net(x, length).shape
    # )

    torch.save(net.state_dict(), os.path.join(ARGS.PROJECTION_PATH, 'save', 'checkpoints', 'demo.pkl'))

    print("Done")
