from torch import nn
from transformers.models.wavlm import WavLMModel, WavLMPreTrainedModel, WavLMConfig, WAVLM_PRETRAINED_MODEL_ARCHIVE_LIST
from transformers.models.wavlm.modeling_wavlm import _compute_mask_indices, WavLMEncoder, WavLMFeatureProjection, \
    WavLMFeatureEncoder

from models.wav2vec2_components import prepare_mask, get_feat_extract_output_lengths
import torch
from torch.nn import functional as F
from main.opts import ARGS


class WavLMWrapper(nn.Module):

    def __init__(self, config):
        super(WavLMWrapper, self).__init__()
        self.config = config
        self.wavlm = WavLMModel.from_pretrained(ARGS.MODEL_NAME_WavLM)

    def forward(self, x, length=None):
        with torch.no_grad():
            x = self.wavlm.feature_extractor(x)
            x = x.transpose(1, 2)
            x, _ = self.wavlm.feature_projection(x)
            mask = None
            if length is not None:
                length = get_feat_extract_output_lengths(length)
                mask = prepare_mask(length, x.shape[:2], x.dtype, x.device)
            if self.training:
                batch_size, sequence_length, hidden_size = x.size()

                # apply SpecAugment along time axis
                if self.config.mask_time_prob > 0:
                    mask_time_indices = _compute_mask_indices(
                        (batch_size, sequence_length),
                        self.config.mask_time_prob,
                        self.config.mask_time_length,
                        min_masks=2
                    )
                    mask_time_indices = torch.tensor(mask_time_indices, device=mask.device)
                    masked_indicies = mask_time_indices & mask
                    flip_mask = torch.rand((batch_size, sequence_length),
                                           device=masked_indicies.device) > 0.0
                    x[masked_indicies & flip_mask] = self.wavlm.masked_spec_embed.to(x.dtype)

                # apply SpecAugment along feature axis
                if self.config.mask_feature_prob > 0:
                    mask_feature_indices = _compute_mask_indices(
                        (batch_size, hidden_size),
                        self.config.mask_feature_prob,
                        self.config.mask_feature_length,
                        min_masks=1
                    )
                    mask_feature_indices = torch.tensor(mask_feature_indices)
                    x[mask_feature_indices[:, None].expand(-1, sequence_length, -1)] = 0
        x = self.wavlm.encoder(x, attention_mask=mask)[0]
        reps = F.relu(x)
        return reps

    def trainable_parameters(self):
        return list(self.wavlm.encoder.parameters())

    def reset_parameters(self):
        pass


class WavLMEncoderWrapper(nn.Module):

    def __init__(self, config: WavLMConfig):
        super().__init__()

        # model only needs masking vector if mask prob is > 0.0
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())
        self.config = config
        self.encoder = WavLMEncoder(self.config)

    def forward(self, x, length):
        """
        :param x:hidden_states
        :param length:
        :return:
        """
        with torch.no_grad():

            mask = None
            if length is not None:
                length = get_feat_extract_output_lengths(length)
                mask = prepare_mask(length, x.shape[:2], x.dtype, x.device)
            if self.training:
                batch_size, sequence_length, hidden_size = x.size()

                # apply SpecAugment along time axis
                if self.config.mask_time_prob > 0:
                    mask_time_indices = _compute_mask_indices(
                        (batch_size, sequence_length),
                        self.config.mask_time_prob,
                        self.config.mask_time_length,
                        min_masks=2
                    )
                    mask_time_indices = torch.tensor(mask_time_indices, device=mask.device)
                    masked_indicies = mask_time_indices & mask
                    flip_mask = torch.rand((batch_size, sequence_length),
                                           device=masked_indicies.device) > 0.0
                    x[masked_indicies & flip_mask] = self.masked_spec_embed.to(x.dtype)

                # apply SpecAugment along feature axis
                if self.config.mask_feature_prob > 0:
                    mask_feature_indices = _compute_mask_indices(
                        (batch_size, hidden_size),
                        self.config.mask_feature_prob,
                        self.config.mask_feature_length,
                        min_masks=1
                    )
                    mask_feature_indices = torch.tensor(mask_feature_indices)
                    x[mask_feature_indices[:, None].expand(-1, sequence_length, -1)] = 0
        x = self.encoder(x, attention_mask=mask)[0]
        reps = F.relu(x)
        return reps

    def trainable_parameters(self):
        return list(self.encoder.parameters()) + list([self.masked_spec_embed])
