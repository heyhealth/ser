"""
AIO -- All Model in One
The original model
from
SPEECH EMOTION RECOGNITION WITH CO-ATTENTION BASED MULTI-LEVEL ACOUSTIC INFORMATION
"""
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Wav2Vec2Model
from models.DeepLearningModels import SER_AlexNet
from main.opts import ARGS
from mydatasets.feature_acoustic import extract_mfcc_from_wavefrom, extract_melspectrogram_from_waveform


# __all__ = ['Wav2EmoNet']
class Wav2Vec2EmoNet(nn.Module):
    def __init__(self, num_classes):
        super(Wav2Vec2EmoNet, self).__init__()

        # CNN for Spectrogram
        self.alexnet_model = SER_AlexNet(num_classes=num_classes, in_ch=3, pretrained=True)

        self.post_spec_dropout = nn.Dropout(p=0.1)
        self.post_spec_layer = nn.Linear(9216, 128)  # 9216 for cnn, 32768 for ltsm s, 65536 for lstm l

        # LSTM for MFCC
        self.lstm_mfcc = nn.LSTM(input_size=40, hidden_size=256, num_layers=2, batch_first=True, dropout=0.5,
                                 bidirectional=True)  # bidirectional = True

        self.post_mfcc_dropout = nn.Dropout(p=0.1)
        # self.post_mfcc_layer = nn.Linear(153600,
        #                                  128)  # 40 for attention and 8064 for conv, 32768 for cnn-lstm, 38400 for lstm
        self.post_mfcc_layer = nn.Linear(154112,
                                         128)  # 40 for attention and 8064 for conv, 32768 for cnn-lstm, 38400 for lstm

        # Spectrogram + MFCC
        self.post_spec_mfcc_att_dropout = nn.Dropout(p=0.1)
        self.post_spec_mfcc_att_layer = nn.Linear(256, 149)  # 9216 for cnn, 32768 for ltsm s, 65536 for lstm l

        # WAV2VEC 2.0
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained(ARGS.MODEL_NAME_Wav2Vec2)

        self.post_wav_dropout = nn.Dropout(p=0.1)
        self.post_wav_layer = nn.Linear(768, 128)  # 512 for 1 and 768 for 2

        # Combination
        self.post_att_dropout = nn.Dropout(p=0.1)
        self.post_att_layer_1 = nn.Linear(384, 128)
        self.post_att_layer_2 = nn.Linear(128, 128)
        self.post_att_layer_3 = nn.Linear(128, num_classes)

    def _compute_MFCC_feature(self, x):
        """
        :param x:[B,waveform]
        :return: MFCC:[B,20,L]
        """
        with torch.no_grad():
            waveform = x[0]
            mfcc = extract_mfcc_from_wavefrom(waveform.cpu().numpy(), sr=16000, n_fft=640,
                                              hop_length=160, fmax=8000)
            mfcc_ = torch.as_tensor(mfcc, dtype=x.dtype, device=torch.device('cpu')).unsqueeze(0)
            for i in range(x.shape[0] - 1):
                mfcc = extract_mfcc_from_wavefrom(waveform.cpu().numpy(), sr=16000, n_fft=640,
                                                  hop_length=160, fmax=8000)
                mfcc_ = np.concatenate([mfcc_, torch.as_tensor(mfcc, dtype=x.dtype, device=torch.device('cpu')).unsqueeze(0)],
                                       axis=0)
        return torch.tensor(mfcc_, dtype=x.dtype, device=x.device)

    def _compute_Mel_Spectrogram_feature(self, x):
        """
        :param x: [B,waveform]
        :return: Mel_Spectrogram:[B,128,L]
        """
        with torch.no_grad():
            waveform = x[0]
            mel_spec = extract_melspectrogram_from_waveform(waveform.cpu().numpy(), sr=16000, n_fft=512,
                                                            hop_length=160, fmax=8000)
            mel_spec = torch.as_tensor(mel_spec, dtype=x.dtype, device=torch.device('cpu')).unsqueeze(0).unsqueeze(0).numpy()
            mel_spec = _spec_to_rgb(mel_spec)
            mel_spec_ = torch.as_tensor(mel_spec, dtype=x.dtype, device=torch.device('cpu')).unsqueeze(0)
            for i in range(x.shape[0] - 1):
                mel_spec = extract_melspectrogram_from_waveform(waveform.cpu().numpy(), sr=16000, n_fft=512,
                                                                hop_length=160, fmax=8000)
                mel_spec = torch.as_tensor(mel_spec, dtype=x.dtype, device=torch.device('cpu')).unsqueeze(0).unsqueeze(0).numpy()
                mel_spec = _spec_to_rgb(mel_spec)
                mel_spec_ = np.concatenate(
                    [mel_spec_, torch.as_tensor(mel_spec, dtype=x.dtype, device=torch.device('cpu')).unsqueeze(0)],
                    axis=0)
        return torch.tensor(mel_spec_, dtype=x.dtype, device=x.device)

    def forward(self, x, length=None):
        # audio_spec: [batch, 3, 256, 384] -> [batch, 3, 224, 526]
        # audio_mfcc: [batch, 300, 40] -> [batch, 301, 40]
        # audio_wav: [32, 48000]
        with torch.no_grad():
            audio_spec = self._compute_Mel_Spectrogram_feature(x)
            audio_mfcc = self._compute_MFCC_feature(x)
            audio_mfcc = audio_mfcc.permute([0, 2, 1])  # B,L(313),C(20)
            audio_wav = x

        audio_mfcc = F.normalize(audio_mfcc, p=2, dim=2)

        # spectrogram - SER_CNN
        audio_spec, output_spec_t = self.alexnet_model(audio_spec)  # [batch, 256, 6, 6], []
        audio_spec = audio_spec.reshape(audio_spec.shape[0], audio_spec.shape[1], -1)  # [batch, 256, 36]

        # audio -- MFCC with BiLSTM
        audio_mfcc, _ = self.lstm_mfcc(audio_mfcc)  # [batch, 300, 512]

        audio_spec_ = torch.flatten(audio_spec, 1)  # [batch, 9216]
        audio_spec_d = self.post_spec_dropout(audio_spec_)  # [batch, 9216]
        audio_spec_p = F.relu(self.post_spec_layer(audio_spec_d), inplace=False)  # [batch, 128]

        # + audio_mfcc = self.att(audio_mfcc)
        audio_mfcc_ = torch.flatten(audio_mfcc, 1)  # [batch, 153600]
        audio_mfcc_att_d = self.post_mfcc_dropout(audio_mfcc_)  # [batch, 153600]
        audio_mfcc_p = F.relu(self.post_mfcc_layer(audio_mfcc_att_d), inplace=False)  # [batch, 128]

        # FOR WAV2VEC2.0 WEIGHTS
        spec_mfcc = torch.cat([audio_spec_p, audio_mfcc_p], dim=-1)  # [batch, 256]
        audio_spec_mfcc_att_d = self.post_spec_mfcc_att_dropout(spec_mfcc)  # [batch, 256]
        audio_spec_mfcc_att_p = F.relu(self.post_spec_mfcc_att_layer(audio_spec_mfcc_att_d),
                                       inplace=False)  # [batch, 149]
        audio_spec_mfcc_att_p = audio_spec_mfcc_att_p.reshape(audio_spec_mfcc_att_p.shape[0], 1, -1)  # [batch, 1, 149]
        # + audio_spec_mfcc_att_2 = F.softmax(audio_spec_mfcc_att_1, dim=2)

        # wav2vec 2.0
        # audio_wav = self.wav2vec2_model(audio_wav.cuda()).last_hidden_state  # [batch, 149, 768]
        audio_wav = self.wav2vec2_model(audio_wav).last_hidden_state  # [batch, 149, 768]
        audio_wav = torch.matmul(audio_spec_mfcc_att_p, audio_wav)  # [batch, 1, 768]
        audio_wav = audio_wav.reshape(audio_wav.shape[0], -1)  # [batch, 768]
        # audio_wav = torch.mean(audio_wav, dim=1)

        audio_wav_d = self.post_wav_dropout(audio_wav)  # [batch, 768]
        audio_wav_p = F.relu(self.post_wav_layer(audio_wav_d), inplace=False)  # [batch, 768]

        ## combine()
        audio_att = torch.cat([audio_spec_p, audio_mfcc_p, audio_wav_p], dim=-1)  # [batch, 384]
        audio_att_d_1 = self.post_att_dropout(audio_att)  # [batch, 384]
        audio_att_1 = F.relu(self.post_att_layer_1(audio_att_d_1), inplace=False)  # [batch, 128]
        audio_att_d_2 = self.post_att_dropout(audio_att_1)  # [batch, 128]
        audio_att_2 = F.relu(self.post_att_layer_2(audio_att_d_2), inplace=False)  # [batch, 128]
        output_att = self.post_att_layer_3(audio_att_2)  # [batch, 4]

        output = {
            'F1': audio_wav_p,
            'F2': audio_att_1,
            'F3': audio_att_2,
            'F4': output_att,
            'M': output_att
        }

        return output['M']

    def trainable_parameters(self):
        return list(self.parameters())


from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _spec_to_rgb(data):
    """
    Convert normalized spectrogram to pseudo-RGB image based on pyplot color map
        and apply AlexNet image pre-processing

    Input: data
            - shape (N,C,H,W) = (num_spectrogram_segments, 1, Freq, Time)
            - data range [0.0, 1.0]
    """

    # AlexNet preprocessing
    alexnet_preprocess = transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get the color map to convert normalized spectrum to RGB
    cm = plt.get_cmap('jet')  # brg #gist_heat #brg #bwr

    # Flip the frequency axis to orientate image upward, remove Channel axis
    data = np.flip(data, axis=2)

    data = np.squeeze(data, axis=1)

    data_tensor = list()

    for i, seg in enumerate(data):
        seg = np.clip(seg, 0.0, 1.0)
        seg_rgb = (cm(seg)[:, :, :3] * 255.0).astype(np.uint8)

        img = Image.fromarray(seg_rgb, mode='RGB')

        data_tensor.append(alexnet_preprocess(img))

    return data_tensor[0]


if __name__ == '__main__':
    """
    x = torch.randn((4, 1, 338, 338)).numpy()

    print(
        _spec_to_rgb(data=x)[1].shape
    )
    """

    # the waveform clip length is 3 seconds
    # lr is set 1e-5
    # use adamW
    # batch is set 64

    x = torch.randn((4, 48000))
    length = torch.tensor([128000, 12300, 12300, 1234], dtype=torch.long)
    net = Wav2Vec2EmoNet(num_classes=12)
    print(
        net(x, length).shape
    )
