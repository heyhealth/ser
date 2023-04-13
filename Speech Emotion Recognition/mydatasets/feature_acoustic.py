import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


# import torch
# from torchvision.transforms import transforms


def extract_waveform_from_wav(wav_path, max_sequence_length=None, padding_values=0, sr=16000):
    """
    load the waveform from given wav path , if given the max_sequence_length , the waveform will be clipped
    loaded by librosa toolkits
    :param wav_path:
    :param max_sequence_length: the clip sequence length (second)
    :param padding:
    :return:
    """
    y, sr = librosa.load(wav_path, sr=sr)
    if max_sequence_length is None:
        return y
    if max_sequence_length * sr > y.shape[0]:
        return np.pad(y, (0, max_sequence_length * sr - y.shape[0]), mode='constant',
                      constant_values=(0, padding_values)), y.shape[0]
    else:
        return np.array(y[0:max_sequence_length * sr]), max_sequence_length * sr


def extract_melspectrogram_from_waveform(waveform, sr, **kwargs):
    """
    :param waveform:
    :param sr: the waveform's sr
    :param kwargs: librosa.feature.melspectrogram(**kwargs)
    :return:
    """
    mel_spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sr, **kwargs)
    return mel_spectrogram


def extract_melspectrogram_from_wav(wav_path, sr, **kwargs):
    """
    :param wav_path:
    :param sr: resample sr , if None , the sr is the original wav's  sr
    :param kwargs:
    :return:
    """
    y, sr = librosa.load(wav_path, sr=sr)
    # Passing through arguments to the Mel filters
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, **kwargs)
    return mel_spectrogram


def Visualize_the_Mel_Frequency_Spectrogram_series(S, sr):
    """
    Display of mel-frequency spectrogram coefficients, with custom
    arguments for mel filterbank construction (default is fmax=sr/2):
    :param S :  np.ndarray [shape=(..., n_mels, t)] Mel spectrogram
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()


def extract_mfcc_from_wavefrom(waveform, sr, **kwargs):
    """
    :param wav_path:
    :param sr: the waveform's sr
    :param kwargs: librosa.feature.melspectrogram(kwargs)
    :return:
    """
    S = librosa.feature.melspectrogram(y=waveform, sr=sr, **kwargs)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)
    return mfcc


def extract_mfcc_from_wav(wav_path, sr, **kwargs):
    """
   :param wav_path:
   :param sr: resample sr , if None , the sr is the original wav's  sr
   :param kwargs:
   :return:
   """
    y, sr = librosa.load(wav_path, sr=sr)
    S = librosa.feature.melspectrogram(y=waveform, sr=sr, **kwargs)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=20)
    return mfcc


def Visualize_the_MFCC_series(mfcc):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title='MFCC')
    plt.show()


def Visualize_the_Mel_and_MFCC_series(S, mfcc):
    fig, ax = plt.subplots(nrows=2, sharex=True)
    img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', fmax=8000, ax=ax[0])
    fig.colorbar(img, ax=[ax[0]])
    ax[0].set(title='Mel spectrogram')
    ax[0].label_outer()
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax[1])
    fig.colorbar(img, ax=[ax[1]])
    ax[1].set(title='MFCC')
    plt.show()


def Compare_different_DCT_bases(y, sr):
    m_slaney = librosa.feature.mfcc(y=y, sr=sr, dct_type=2)
    m_htk = librosa.feature.mfcc(y=y, sr=sr, dct_type=3)
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
    img1 = librosa.display.specshow(m_slaney, x_axis='time', ax=ax[0])
    ax[0].set(title='RASTAMAT / Auditory toolbox (dct_type=2)')
    fig.colorbar(img1, ax=[ax[0]])
    img2 = librosa.display.specshow(m_htk, x_axis='time', ax=ax[1])
    ax[1].set(title='HTK-style (dct_type=3)')
    fig.colorbar(img2, ax=[ax[1]])
    plt.show()


def test_Visualize():
    wav_path = r"J:\Datasets\DAIC\DAIC-WOZ-WAVS-PARTICIPANT\300_PARTICIPANT.wav"
    y, sr = librosa.load(wav_path)
    S = extract_melspectrogram_from_wav(wav_path)
    mfcc = extract_mfcc_from_wav(wav_path)
    Visualize_the_Mel_Frequency_Spectrogram_series(S=S, sr=16000)
    Visualize_the_MFCC_series(mfcc)
    Visualize_the_Mel_and_MFCC_series(S, mfcc)
    Compare_different_DCT_bases(y, sr)
