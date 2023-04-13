import math

import torch
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
from main.opts import ARGS


# Modified from huggingface
def get_feat_extract_output_lengths(input_length, conv_kernel=(10, 3, 3, 3, 3, 2, 2),
                                    conv_stride=(5, 2, 2, 2, 2, 2, 2)):
    """
    Computes the output length of the convolutional layers
    """

    def _conv_out_length(input_length, kernel_size, stride):
        # 1D convolutional layer output length formula taken
        # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        return (input_length - kernel_size) // stride + 1

    for kernel_size, stride in zip(conv_kernel, conv_stride):
        input_length = _conv_out_length(input_length, kernel_size, stride)
    return input_length


# Modified from huggingface
def prepare_mask(length, shape, dtype, device):
    mask = torch.zeros(
        shape, dtype=dtype, device=device
    )
    # these two operations makes sure that all values
    # before the output lengths indices are attended to
    mask[
        (torch.arange(mask.shape[0], device=device), length - 1)
    ] = 1
    mask = mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    return mask


def get_pretrained_wav2vec2_model():
    """
    the wav2vec2 model which pretrained (facebook/wav2vec2-base-960h)
    :return:
    """
    pretrained_wav2vec2_model = Wav2Vec2Model.from_pretrained(ARGS.MODEL_NAME_Wav2Vec2)
    return pretrained_wav2vec2_model


def get_pretrained_wav2vec2_model_config():
    """
    the wav2vec2 model config which pretrained (facebook/wav2vec2-base-960h)
    :return:
    """
    pretrained_wav2vec2_model = Wav2Vec2Model.from_pretrained(ARGS.MODEL_NAME_Wav2Vec2)
    return pretrained_wav2vec2_model.config


def global_avg_pooling1d_with_masks(hidden_states, length, conv_kernel=(10, 3, 3, 3, 3, 2, 2),
                                    conv_stride=(5, 2, 2, 2, 2, 2, 2), return_mask=False):
    """
    :param hidden_states: [B,L,C]
    :param length:
    :param conv_kernel:
    :param conv_stride:
    :return:
    """
    last_feat_pos = get_feat_extract_output_lengths(length, conv_kernel=conv_kernel,
                                                    conv_stride=conv_stride) - 1
    logits = hidden_states.permute(1, 0, 2)  # L, B, C
    masks = torch.arange(logits.size(0), device=logits.device).expand(last_feat_pos.size(0),
                                                                      -1) < last_feat_pos.unsqueeze(1)
    masks = masks.float()
    embedding = (logits * masks.T.unsqueeze(-1)).sum(0) / last_feat_pos.unsqueeze(1)
    if return_mask:
        return embedding, (masks, last_feat_pos)
    return embedding


def global_avg_pooling1d_with_masks_for_acoustic(hidden_states, length, hop_length):
    """
    :param hidden_states: [B,L,C]
    :param length:
    :param conv_kernel:
    :param conv_stride:
    :return:
    """
    last_feat_pos = length // hop_length
    logits = hidden_states.permute(1, 0, 2)  # L, B, C
    masks = torch.arange(logits.size(0), device=logits.device).expand(last_feat_pos.size(0),
                                                                      -1) < last_feat_pos.unsqueeze(1)
    masks = masks.float()
    embedding = (logits * masks.T.unsqueeze(-1)).sum(0) / last_feat_pos.unsqueeze(1)
    return embedding


if __name__ == '__main__':
    model = get_pretrained_wav2vec2_model()
    print(
        model.config
    )
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Config

    config2 = Wav2Vec2Config()
    print(
        config2
    )
