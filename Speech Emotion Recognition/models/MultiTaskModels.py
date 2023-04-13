import os, sys

sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))
import collections
from torch import nn
import torch
from models.wav2vec2_conv_modules4 import Wav2Vec2EncoderWrapper
from models.wav2vec2_components import get_feat_extract_output_lengths, get_pretrained_wav2vec2_model, \
    get_pretrained_wav2vec2_model_config
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Config, Wav2Vec2FeatureProjection, Wav2Vec2ForCTC
from transformers.models.wavlm.modeling_wavlm import WavLMConfig
from models.wav2vec2_conv_modules2 import Wav2Vec2FeatureEncoder
from models.wavlm_wrappers import WavLMWrapper
from main.opts import ARGS

from utils.pooling_v3 import IndexPool1D


# first & cls
class Wav2Vec2EmoNet(nn.Module):

    def __init__(self, num_classes, vocab_size):
        super(Wav2Vec2EmoNet, self).__init__()
        # Wav2Vec2 config
        self.config = Wav2Vec2Config(
            vocab_size=vocab_size,
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
        self.insert_cls_token = True
        self.cls_token_constant = 1

        # Wav2Vec2 feature_extractor
        self.feature_extractor = Wav2Vec2FeatureEncoder(self.config)

        # Wav2Vec2 feature_projection
        self.feature_projection = Wav2Vec2FeatureProjection(self.config)

        # Wav2Vec2 transformer encoder
        self.encoder = Wav2Vec2EncoderWrapper(config=self.config)

        self.pooling = IndexPool1D(
            selection_method="first+cls", dim_to_reduce=1
        )

        # classifier head for ser task
        self.classifier_ser = nn.Sequential(
            nn.ReLU(),
            nn.Linear(768, num_classes)
        )
        # classifier head for asr task
        self.classifier_asr = nn.Sequential(
            nn.ReLU(),
            nn.Linear(768, vocab_size)
        )

        # initialize model parameters
        self.reset_parameters()

    def forward(self, x, length, labels=None):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
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
        """
        # ----- global avg pooling with masks -----
        last_feat_pos = get_feat_extract_output_lengths(length, conv_kernel=self.config.conv_kernel,
                                                        conv_stride=self.config.conv_stride) - 1
        logits = hidden_states.permute(1, 0, 2)  # L, B, C
        masks = torch.arange(logits.size(0), device=logits.device).expand(last_feat_pos.size(0),
                                                                          -1) < last_feat_pos.unsqueeze(1)
        masks = masks.float()
        embedding = (logits * masks.T.unsqueeze(-1)).sum(0) / last_feat_pos.unsqueeze(1)
        """
        embedding = self.pooling(hidden_states)
        # ----- classifier head -----
        logits_ser = self.classifier_ser(embedding)
        # ASR task using the wav2vec2 encoder's outputs (hidden_states without cls token)
        hidden_states_used_by_asr = hidden_states[:, 1:, :]
        logits_asr = self.classifier_asr(hidden_states_used_by_asr)

        loss = None
        if labels is not None:

            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths
            input_lengths = get_feat_extract_output_lengths(length).to(torch.long)
            # when not being attended to
            # assuming that padded tokens are filled with -100
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits_asr, dim=-1, dtype=torch.float32).transpose(0, 1)
            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction='mean',
                    zero_infinity=True,
                )
            return logits_ser, loss

        return logits_ser

    def reset_parameters(self):
        """initialize the modules' parameters"""
        pretrained_wav2vec2_model = get_pretrained_wav2vec2_model()
        pretrained_feature_extractor_state_dict = pretrained_wav2vec2_model.feature_extractor.state_dict()
        pretrained_feature_projection_state_dict = pretrained_wav2vec2_model.feature_projection.state_dict()
        self.feature_extractor.load_state_dict(pretrained_feature_extractor_state_dict)
        self.feature_projection.load_state_dict(pretrained_feature_projection_state_dict)

    def trainable_parameters(self):
        params = list(self.encoder.trainable_parameters()) + list(
            self.classifier_asr.parameters()) + list(self.classifier_ser.parameters())
        return params

    def asr_task_trainable_parameters(self):
        params = list(self.encoder.trainable_parameters()) + list(self.classifier_asr.parameters())
        return params

    def ser_task_trainable_parameters(self):
        params = list(self.encoder.trainable_parameters()) + list(self.classifier_ser.parameters())
        return params


class EmoNet_WavLM(nn.Module):

    def __init__(self, num_classes):
        super(EmoNet_WavLM, self).__init__()
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
        self.wavlm = WavLMWrapper(config=self.config)
        # classifier head
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(768, num_classes)
        )

    def forward(self, x, length):
        # B,T -> B,C,T

        # ----- wavlm -----
        hidden_states = self.wavlm(x, length)
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

    def trainable_parameters(self):
        return self.wavlm.trainable_parameters() + list(self.classifier.parameters())


if __name__ == '__main__':
    """
    # wav2vec2 = Wav2Vec2Model.from_pretrained(ARGS.MODEL_NAME_Wav2Vec2)
    # print(
    #     wav2vec2
    # )
    """
    from transformers.models.wav2vec2 import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC

    x = torch.randn(4, 128000)
    length = torch.tensor([128000, 12300, 12300, 1234], dtype=torch.long)
    token_ids = torch.arange(1, 21).reshape((4, 5)).long()
    net = Wav2Vec2EmoNet(num_classes=5, vocab_size=1234)
    # net = Wav2Vec2ForCTC.from_pretrained(ARGS.MODEL_NAME_Wav2Vec2)
    print(
        net(x, length, token_ids)
        # print(net)
    )
