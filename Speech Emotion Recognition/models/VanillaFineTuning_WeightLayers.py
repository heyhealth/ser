import collections
from torch import nn
from torch.nn import functional as F
import torch
from models.wav2vec2_conv_modules4 import Wav2Vec2EncoderWrapper
from models.wav2vec2_components import get_pretrained_wav2vec2_model, global_avg_pooling1d_with_masks, \
    get_feat_extract_output_lengths
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Config, Wav2Vec2FeatureProjection
from models.wav2vec2_conv_modules2 import Wav2Vec2FeatureEncoder
from utils.pooling_v2 import SelfAttentionPooling, FrameAttentivePooling
from main.opts import ARGS


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
        # the encoder layers' weights
        """
        self.layers_weights = nn.Parameter(torch.randn((1, self.config.num_hidden_layers + 1), dtype=torch.float32),
                                           requires_grad=True)
        """
        # the another implement method (weighted sum all layers)
        num_layers = self.config.num_hidden_layers + 1
        self.layers_weights = nn.Parameter(torch.ones(num_layers) / num_layers, requires_grad=True)

        # pooling method for emotion recognition
        # self.pooling = SelfAttentionPooling(input_dim=self.config.hidden_size)
        self.pooling = FrameAttentivePooling(hidden_states_dim=768, pooling_type='sum')
        # initialize model parameters
        self.reset_parameters()

    def forward(self, x, length):
        # B,T
        with torch.no_grad():
            x = self.feature_extractor(x)
            x = x.transpose(1, 2)
            hidden_states, norm_hidden_states = self.feature_projection(x)

        # ----- wav2vec2 encoder -----
        hidden_states_after_relu, all_hidden_states, all_self_attentions = self.encoder(hidden_states,
                                                                                        length,
                                                                                        return_all_hidden_states=True)
        # ----- stack hidden_states -----
        all_hidden_states_list = []
        for hidden_states_ in all_hidden_states:
            all_hidden_states_list.append(hidden_states_.unsqueeze(0))
        all_hidden_states_stacks = torch.concat(all_hidden_states_list, dim=0)  # torch.Size([13, 4, 399, 768])
        # add layer_norm on all_hidden_states_ (2022.12.6)
        all_hidden_states_stacks = F.layer_norm(all_hidden_states_stacks, (all_hidden_states_stacks.shape[-1],))
        # add relu (2022.12.7)
        # torch.Size([13, 4, 349, 768])
        all_hidden_states_stacks = F.relu(all_hidden_states_stacks)
        # ----- weighted sum all layers -----
        _, *original_shape = all_hidden_states_stacks.shape
        # torch.Size([13, 1072128])
        all_hidden_states_stacks = all_hidden_states_stacks.view(self.config.num_hidden_layers + 1, -1)
        norm_layers_weights = F.softmax(self.layers_weights, dim=-1)
        # torch.Size([1072128])
        weighted_hidden_states = (norm_layers_weights.unsqueeze(-1) * all_hidden_states_stacks).sum(dim=0)
        # torch.Size([4, 349, 768])
        weighted_hidden_states = weighted_hidden_states.view(*original_shape)
        """
        # self.layers_weights.shape torch.Size([1, 13])
        weighted_hidden_states = torch.mul(F.softmax(self.layers_weights, dim=-1),
                                           all_hidden_states_stacks.permute([1, 2, 3, 0])).sum(dim=-1)
        """
        # the all_hidden_states includes hidden_states but no relu
        hidden_states = weighted_hidden_states
        """
        # ----- global avg pooling with masks -----
        embedding = global_avg_pooling1d_with_masks(hidden_states, length, self.config.conv_kernel,
                                                    self.config.conv_stride)
        """
        # ----- SelfAttentionPooling with masks -----
        _, (mask, last_feat_pos) = global_avg_pooling1d_with_masks(hidden_states, length, self.config.conv_kernel,
                                                                   self.config.conv_stride, return_mask=True)
        embedding = self.pooling(hidden_states, attention_mask=(mask, last_feat_pos))
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
        params = list(self.feature_projection.parameters()) + \
                 list(self.encoder.trainable_parameters()) + \
                 list(self.classifier.parameters()) + \
                 list([self.layers_weights]) + \
                 list(self.pooling.parameters())
        return params


if __name__ == '__main__':
    """
    # wav2vec2 = Wav2Vec2Model.from_pretrained(ARGS.MODEL_NAME_Wav2Vec2)
    # print(
    #     wav2vec2
    # )
    """
    x = torch.randn(4, 112000)
    length = torch.tensor([112000, 12300, 12300, 1234], dtype=torch.long)
    net = Wav2Vec2EmoNet(num_classes=12)
    print(
        net(x, length).shape
    )
    """
    x = torch.randn(4, 399, 768)
    print(
        global_avg_pooling1d_with_masks(x, length).shape
    )
    """
