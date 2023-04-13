from torch import nn
import torch
from typing import List, Tuple
from torch.nn import functional as F


class Wav2Vec2FeatureEncoderWrapper(nn.Module):
    """ extract feature from raw waveform """

    def __init__(self):
        super(Wav2Vec2FeatureEncoderWrapper, self).__init__()
        arch = "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]"
        self.feature_extractor = ConvFeatureExtractionModel(conv_layers=eval(arch), mode='layer_norm')
        self.feature_projection = Wav2Vec2FeatureProjection()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x, _ = self.feature_projection(x)
        return x


# Modified transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureProjection
class Wav2Vec2FeatureProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm(512, eps=1e-05, elementwise_affine=True)
        self.projection = nn.Linear(in_features=512, out_features=768, bias=True)
        self.dropout = nn.Dropout(p=0.1, inplace=False)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


# Copied from fairseq's wav2vec2 ConvFeatureExtractionModel
class ConvFeatureExtractionModel(nn.Module):

    def __init__(
            self,
            conv_layers: List[Tuple[int, int, int]],
            dropout: float = 0.0,
            mode: str = "default",
            conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
                n_in,
                n_out,
                k,
                stride,
                is_layer_norm=False,
                is_group_norm=False,
                conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                           is_layer_norm and is_group_norm
                   ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x

# Copied from fairseq's wav2vec2
class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)

# Copied from fairseq's wav2vec2
class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


# Copied from fairseq's wav2vec2
class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


if __name__ == '__main__':
    """
       # usage
       feature_enc_layers = "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]"
       self.feature_extractor = ConvFeatureExtractionModel(conv_layers=eval(feature_enc_layers))
       """

    x = torch.randn((1, 12800))
    net = Wav2Vec2FeatureEncoderWrapper()
    print(
        net(x).shape
    )
