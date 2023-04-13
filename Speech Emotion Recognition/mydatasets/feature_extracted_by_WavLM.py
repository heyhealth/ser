import os
import sys

sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model, WavLMModel
import torchaudio
from torchaudio.backend.soundfile_backend import load
from torch import nn
from main.opts import ARGS
import librosa

PROCESSOR = Wav2Vec2Processor.from_pretrained(ARGS.MODEL_NAME_WavLM)
MODEL = WavLMModel.from_pretrained(ARGS.MODEL_NAME_WavLM)


def global_avg_pooling(x):
    return torch.mean(x, dim=1)


def extract_feature_by_WavLM(wav_path, resampled=False):
    """
    extract the features from wav using pretrained model WavLM
    :param wav_path:
    :return:
    """
    if resampled is True:
        y, sr = librosa.load(wav_path, sr=16000)
    else:
        y, sr = load(wav_path)
        y = y.squeeze()
    inputs = PROCESSOR(y, sampling_rate=PROCESSOR.feature_extractor.sampling_rate, return_tensors="pt",
                       padding=True).input_values
    inputs = inputs.to(ARGS.DEVICE)
    MODEL.to(ARGS.DEVICE)
    with torch.no_grad():
        outputs = MODEL(inputs)
    # Sequence of extracted feature vectors of the last convolutional layer of the model.
    features_ = outputs['extract_features']
    # Sequence of hidden-states at the output of the last layer of the model.
    # features_ = outputs['last_hidden_state']
    """
    model_name："facebook/wav2vec2-base-960h"
    last_hidden_state:768d,extract_features:512d
    """
    with torch.no_grad():
        features_ = features_.permute([0, 2, 1])
        features = nn.AdaptiveAvgPool1d(128)(features_)
        features = features.squeeze()
    return features


def do_test():
    # wav_path = r"E:\Datasets\IEMOCAP\raw_unzip\Session5\sentences\wav\Ses05M_impro07\Ses05M_impro07_F031.wav"
    wav_path = r"E:\Datasets\RAVDESS\raw\Audio-only-files\Audio_Song_Actors_01-24\Actor_01\03-02-05-02-02-02-01.wav"
    feat = extract_feature_by_WavLM(wav_path, resampled=True)
    print(feat.shape)


if __name__ == '__main__':
    do_test()
