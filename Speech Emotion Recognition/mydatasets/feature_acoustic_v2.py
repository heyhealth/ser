import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import numpy as np
import torch
"""
Audio-and-text-based-emotion-recognition
A multimodal approach on emotion recognition using audio and text.

A pytorch implementation of the paper

Attention Based Fully Convolutional Network for Speech Emotion Recognition (https://arxiv.org/pdf/1806.01506v2.pdf)
Multimodal Speech Emotion Recognition using Audio and Text (https://arxiv.org/pdf/1810.04635.pdf)
Emotion Recognition from Speech (https://arxiv.org/pdf/1912.10458.pdf)

"""


def audio2spectrogram(filepath):
    # fig = plt.figure(figsize=(5,5))
    samplerate, test_sound = wavfile.read(filepath, mmap=True)
    # print('samplerate',samplerate)
    _, spectrogram = log_specgram(test_sound, samplerate)
    # print(spectrogram.shape)
    # print(type(spectrogram))
    # plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    return spectrogram


def audio2wave(filepath):
    fig = plt.figure(figsize=(5, 5))
    samplerate, test_sound = wavfile.read(filepath, mmap=True)
    plt.plot(test_sound)


def log_specgram(audio, sample_rate, window_size=40,
                 step_size=20, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    # print('noverlap',noverlap)
    # print('nperseg',nperseg)
    freqs, _, spec = signal.spectrogram(audio,
                                        fs=sample_rate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)
    return freqs, np.log(spec.T.astype(np.float32) + eps)



def get_3d_spec(Sxx_in, moments=None):
    if moments is not None:
        (base_mean, base_std, delta_mean, delta_std,
         delta2_mean, delta2_std) = moments
    else:
        base_mean, delta_mean, delta2_mean = (0, 0, 0)
        base_std, delta_std, delta2_std = (1, 1, 1)
    h, w = Sxx_in.shape
    # 将第1个时间步和所有的时间步水平concat 然后取最后一个时间步 作为right1 ？？？ [T,1]
    right1 = np.concatenate([Sxx_in[:, 0].reshape((h, -1)), Sxx_in], axis=1)[:, :-1]
    # delta 为原来的声谱图-去最后一个时间步，取第二个时间步之后的数据 [T,320]
    delta = (Sxx_in - right1)[:, 1:]
    # delta_pad 为上面的数据取第一个时间步 [T,1]
    delta_pad = delta[:, 0].reshape((h, -1))
    # 新的delta为delta重复一下第一个列数据
    delta = np.concatenate([delta_pad, delta], axis=1)
    # 得到第3个通道的方法 和 得到第2个通道的方法 一致
    right2 = np.concatenate([delta[:, 0].reshape((h, -1)), delta], axis=1)[:, :-1]
    delta2 = (delta - right2)[:, 1:]
    delta2_pad = delta2[:, 0].reshape((h, -1))
    delta2 = np.concatenate([delta2_pad, delta2], axis=1)
    # 每个通道进行归一化
    base = (Sxx_in - base_mean) / base_std
    delta = (delta - delta_mean) / delta_std
    delta2 = (delta2 - delta2_mean) / delta2_std
    # 三个通道concat 即可
    stacked = [arr.reshape((h, w, 1)) for arr in (base, delta, delta2)]
    return np.concatenate(stacked, axis=2)



if __name__ == '__main__':
    # wav_path = r"E:\Datasets\IEMOCAP\raw_unzip\Session5\sentences\wav\Ses05F_impro02\Ses05F_impro02_F000.wav"
    # torch.Size([3, 123, 321])
    # wav_path = r"E:\Datasets\IEMOCAP\raw_unzip\Session5\sentences\wav\Ses05F_impro02\Ses05F_impro02_F008.wav"
    # torch.Size([3, 133, 321])
    # wav_path = r"E:\Datasets\IEMOCAP\raw_unzip\Session5\sentences\wav\Ses05F_impro02\Ses05F_impro02_M008.wav"
    # torch.Size([3, 76, 321])
    wav_path = r"E:\Datasets\IEMOCAP\raw_unzip\Session5\sentences\wav\Ses05F_impro02\Ses05F_impro02_M028.wav"
    # torch.Size([3, 596, 321])
    spec = audio2spectrogram(wav_path)

    print(
        spec.shape
    )
    spec_3d = get_3d_spec(spec)

    trans_spec = np.transpose(spec_3d, (2, 0, 1))

    input_tensor = torch.tensor(trans_spec)

    print(input_tensor.shape)

