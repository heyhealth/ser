import collections
from torch import nn
import torch
from models.wav2vec2_conv_modules4 import Wav2Vec2EncoderWrapper
from models.wav2vec2_components import get_feat_extract_output_lengths, get_pretrained_wav2vec2_model, \
    get_pretrained_wav2vec2_model_config
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Config, Wav2Vec2FeatureProjection
from transformers.models.wavlm.modeling_wavlm import WavLMConfig
from models.wav2vec2_conv_modules2 import Wav2Vec2FeatureEncoder
from models.wavlm_wrappers import WavLMWrapper
from main.opts import ARGS

"""
global avg pooling with masks
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

    def forward(self, x, length):
        with torch.no_grad():
            x = self.feature_extractor(x)
            x = x.transpose(1, 2)
            hidden_states, _ = self.feature_projection(x)
        # ----- wav2vec2 encoder -----
        hidden_states = self.encoder(hidden_states, length)
        # ----- global avg pooling with masks -----
        last_feat_pos = get_feat_extract_output_lengths(length, conv_kernel=self.config.conv_kernel,
                                                        conv_stride=self.config.conv_stride) - 1
        logits = hidden_states.permute(1, 0, 2)  # L, B, C
        masks = torch.arange(logits.size(0), device=logits.device).expand(last_feat_pos.size(0),
                                                                          -1) < last_feat_pos.unsqueeze(1)
        masks = masks.float()
        logits = (logits * masks.T.unsqueeze(-1)).sum(0) / last_feat_pos.unsqueeze(1)
        # ----- classifier head -----
        logits = self.classifier(logits)
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


if __name__ == '__main__':
    """
    # wav2vec2 = Wav2Vec2Model.from_pretrained(ARGS.MODEL_NAME_Wav2Vec2)
    # print(
    #     wav2vec2
    # )
    """
    x = torch.randn(4, 128000)
    length = torch.tensor([128000, 12300, 12300, 1234], dtype=torch.long)
    net = Wav2Vec2EmoNet(num_classes=12)
    print(
        net(x, length).shape
        # print(net)
    )
