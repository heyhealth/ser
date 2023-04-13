from transformers.models.wav2vec2 import Wav2Vec2Model, Wav2Vec2ForPreTraining, Wav2Vec2Config,Wav2Vec2ForAudioFrameClassification,Wav2Vec2ForSequenceClassification
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from main.opts import ARGS
from torch import nn
import torch
from torch.nn import functional as F
from models.wav2vec2_components import prepare_mask, get_feat_extract_output_lengths

"""
The bare Wav2Vec2 Model transformer outputting raw hidden-states without any specific head on top.
"""


# for the wav2vec2 V-FT
class Wav2vec2Wrapper(nn.Module):
    def __init__(self, pretrain=True):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(ARGS.MODEL_NAME_Wav2Vec2)
        # self.wav2vec2 = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base", revision='2dcc7b7f9b11f0ef271067e62599a27317a03114').wav2vec2
        # Disable gradient checkpointing for ddp
        self.wav2vec2.encoder.config.gradient_checkpointing = False
        self.pretrain = pretrain
        if pretrain:
            self.mask_time_length = 15
            self.mask_time_prob = 0.06  # Probability of each time step is masked!
            self.observe_time_prob = 0.0  # Percentage of tokens that are perserved
            self.mask_feature_prob = 0
        else:
            # SpecAug
            self.mask_time_length = 15
            self.mask_time_prob = 0.08
            self.observe_time_prob = 0.0

            self.mask_feature_length = 64
            self.mask_feature_prob = 0.05

    def trainable_params(self):
        ret = list(self.wav2vec2.encoder.parameters())
        return ret

    def forward(self, x, length=None):
        with torch.no_grad():
            x = self.wav2vec2.feature_extractor(x)
            x = x.transpose(1, 2)  # New version of huggingface
            x, _ = self.wav2vec2.feature_projection(x)  # New version of huggingface
            mask = None
            if length is not None:
                length = get_feat_extract_output_lengths(length)
                mask = prepare_mask(length, x.shape[:2], x.dtype, x.device)
            if self.pretrain or self.training:
                batch_size, sequence_length, hidden_size = x.size()

                # apply SpecAugment along time axis
                if self.mask_time_prob > 0:
                    mask_time_indices = _compute_mask_indices(
                        (batch_size, sequence_length),
                        self.mask_time_prob,
                        self.mask_time_length,
                        min_masks=2
                    )
                    mask_time_indices = torch.tensor(mask_time_indices, device=mask.device)
                    masked_indicies = mask_time_indices & mask
                    flip_mask = torch.rand((batch_size, sequence_length),
                                           device=masked_indicies.device) > self.observe_time_prob
                    x[masked_indicies & flip_mask] = self.wav2vec2.masked_spec_embed.to(x.dtype)

                # apply SpecAugment along feature axis
                if self.mask_feature_prob > 0:
                    mask_feature_indices = _compute_mask_indices(
                        (batch_size, hidden_size),
                        self.mask_feature_prob,
                        self.mask_feature_length,
                        min_masks=1
                    )
                    mask_feature_indices = torch.tensor(mask_feature_indices)
                    x[mask_feature_indices[:, None].expand(-1, sequence_length, -1)] = 0
        x = self.wav2vec2.encoder(x, attention_mask=mask)[0]
        reps = F.relu(x)
        if self.pretrain:
            return reps, masked_indicies
        return reps


# for the wav2vec2 TAPT v1
class Wav2vec2PretrainWrapper(nn.Module):
    def __init__(self):
        super().__init__()

        self.wav2vec2PT = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base",
                                                                 revision='2dcc7b7f9b11f0ef271067e62599a27317a03114')
        self.wav2vec2 = self.wav2vec2PT.wav2vec2

        # self.wav2vec2PT.freeze_feature_extractor()

    def trainable_params(self):
        ret = list(self.wav2vec2PT.parameters())
        return ret

    def forward(self, x, length=None):
        self.wav2vec2PT.train()
        with torch.no_grad():
            batch_size, sequence_length = x.size()
            sequence_length = get_feat_extract_output_lengths(sequence_length)
            feat_shape = (batch_size, sequence_length)
            length = get_feat_extract_output_lengths(length)
            attn_mask = prepare_mask(length, feat_shape, x.dtype, x.device)
            mask_time_indices = _compute_mask_indices(
                feat_shape,
                self.wav2vec2PT.config.mask_time_prob,
                self.wav2vec2PT.config.mask_time_length,
                min_masks=2,
                device=x.device,
                attention_mask=attn_mask
            )
        x = self.wav2vec2PT(x, mask_time_indices=mask_time_indices)  # , attention_mask=attn_mask)
        return x


# for the wav2vec2 TAPT v2
class Wav2Vec2ForPreTrainingWrapper(nn.Module):

    def __init__(self):
        super(Wav2Vec2ForPreTrainingWrapper, self).__init__()
        self.wav2vec2PT = Wav2Vec2ForPreTraining.from_pretrained(ARGS.MODEL_NAME_Wav2Vec2)
        self.wav2vec2PT.config.mask_time_length = ARGS.AUDIO_SEGMENT_TIME_LENGTH
        self.wav2vec2 = self.wav2vec2PT.wav2vec2

    def forward(self, input_values, length=None):
        self.wav2vec2PT.train()
        with torch.no_grad():
            batch_size, raw_sequence_length = input_values.shape
            sequence_length = get_feat_extract_output_lengths(raw_sequence_length)

            length = get_feat_extract_output_lengths(length)
            attention_mask = prepare_mask(length, (batch_size, sequence_length), input_values.dtype,
                                          input_values.device)

            mask_time_indices = _compute_mask_indices((batch_size, sequence_length),
                                                      mask_prob=self.wav2vec2PT.config.mask_time_prob,
                                                      mask_length=self.wav2vec2PT.config.mask_time_length,
                                                      min_masks=2,
                                                      attention_mask=attention_mask
                                                      )
            sampled_negative_indices = _sample_negative_indices(
                (batch_size, sequence_length), self.wav2vec2PT.config.num_negatives, mask_time_indices
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=input_values.device, dtype=torch.long)
            sampled_negative_indices = torch.tensor(sampled_negative_indices, device=input_values.device,
                                                    dtype=torch.long)

        outputs = self.wav2vec2PT(input_values, mask_time_indices=mask_time_indices,
                                  sampled_negative_indices=sampled_negative_indices)  # ,attention_mask=attention_mask)
        """
           return Wav2Vec2ForPreTrainingOutput(
            loss=loss,
            projected_states=transformer_features,
            projected_quantized_states=quantized_features,
            codevector_perplexity=codevector_perplexity,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            contrastive_loss=contrastive_loss,
            diversity_loss=diversity_loss,
        )
        """
        return outputs

    def trainable_parameters(self):
        return list(self.wav2vec2PT.parameters())

    def reset_parameters(self):
        pass


if __name__ == '__main__':
    x = torch.randn((4, 80000), device='cpu')
    length = torch.tensor([77880] * 4, dtype=torch.long, device='cpu')

    net = Wav2Vec2ForPreTrainingWrapper()
    net.train()
    print(
        net(x, length).loss,
        net(x, length).contrastive_loss,
        net(x, length).diversity_loss
    )
    """
    tensor(373.2771, grad_fn=<AddBackward0>) tensor(371.9341, grad_fn=<NllLossBackward0>) tensor(0.7541, grad_fn=<MulBackward0>)
    """
