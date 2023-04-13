from main.opts import ARGS

from transformers.models.wavlm import WavLMModel


def get_pretrained_wavlm_model():
    """
    the wavlm model which pretrained (models--patrickvonplaten--wavlm-libri-clean-100h-base-plus)
    :return:
    """
    pretrained_wavlm_model = WavLMModel.from_pretrained(ARGS.MODEL_NAME_WavLM)
    return pretrained_wavlm_model


def get_pretrained_wavlm_model_config():
    """
    the wavlm model config which pretrained (models--patrickvonplaten--wavlm-libri-clean-100h-base-plus)
    :return:
    """
    pretrained_wavlm_model = WavLMModel.from_pretrained(ARGS.MODEL_NAME_WavLM)
    return pretrained_wavlm_model.config
