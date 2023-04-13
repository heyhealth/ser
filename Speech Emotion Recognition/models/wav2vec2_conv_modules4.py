from typing import Optional
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Encoder, _compute_mask_indices
import torch
from torch import nn
from torch.nn import functional as F
from models.wav2vec2_components import prepare_mask, get_feat_extract_output_lengths, get_pretrained_wav2vec2_model


class Wav2Vec2EncoderWrapper(nn.Module):

    def __init__(self, config):
        super(Wav2Vec2EncoderWrapper, self).__init__()
        self.config = config
        self.wav2vec2_encoder = Wav2Vec2Encoder(self.config)
        # model only needs masking vector if mask prob is > 0.0
        if self.config.mask_time_prob > 0.0 or self.config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(self.config.hidden_size).uniform_())
        # initialize model parameters
        self.reset_parameters()

    def forward(self, hidden_states, length=None, return_all_hidden_states=False):
        """
        :param x: the feature extracted by Wav2Vec2 feature_extractor
        :param length: used for mask
        :return: the hidden_states
        """

        with torch.no_grad():
            mask = None
            if length is not None:
                length = get_feat_extract_output_lengths(length)
                mask = prepare_mask(length, hidden_states.shape[:2], hidden_states.dtype, hidden_states.device)

            hidden_states = self._mask_hidden_states(hidden_states, mask)
        # the all_hidden_states includes the hidden_states(on the last location)
        hidden_states, all_hidden_states, all_self_attentions = self.wav2vec2_encoder(hidden_states,
                                                                                      attention_mask=mask,
                                                                                      output_attentions=True,
                                                                                      output_hidden_states=True,
                                                                                      return_dict=False
                                                                                      )
        hidden_states = F.relu(hidden_states)
        if return_all_hidden_states:
            return hidden_states, all_hidden_states, all_self_attentions
        return hidden_states

    def reset_parameters(self):
        """initialize the encoder's parameters"""
        pretrained_wav2vec2_model = get_pretrained_wav2vec2_model()
        pretrained_encoder_state_dict = pretrained_wav2vec2_model.encoder.state_dict()
        self.wav2vec2_encoder.load_state_dict(pretrained_encoder_state_dict)

    def trainable_parameters(self):
        return list(self.wav2vec2_encoder.parameters())

    # Modified from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model's _mask_hidden_states
    def _mask_hidden_states(
            self,
            hidden_states: torch.FloatTensor,
            mask,
            mask_time_indices: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """
        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            masked_time_indicies_ = mask_time_indices & mask
            flip_mask = torch.rand((batch_size, sequence_length),
                                   device=masked_time_indicies_.device) > 0.0
            hidden_states[masked_time_indicies_ & flip_mask] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states


if __name__ == '__main__':
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Config

    config = Wav2Vec2Config(
        # num_hidden_layers=4, num_attention_heads=8,
        gradient_checkpointing=False,
        # SpecAugment
        apply_spec_augment=True,
        mask_time_length=15, mask_time_prob=0.08,
        mask_feature_length=64,
        mask_feature_prob=0.05, mask_time_min_masks=2
    )
    x = torch.randn(4, 400, 768)
    length = torch.tensor([12345, 1313, 4124, 12442])
    net = Wav2Vec2EncoderWrapper(config)
    print(
        net(x, length).shape
    )
