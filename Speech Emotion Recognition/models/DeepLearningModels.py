from typing import Type, Union, List, Optional, Any, Callable
import torch
import torchaudio.transforms
import torchvision
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.models import resnet101, ResNet101_Weights, ResNet, ResNet50_Weights, ResNet34_Weights, \
    ResNet18_Weights, resnet18, resnet34
from torchvision.models.resnet import Bottleneck, BasicBlock
from torchvision import transforms
from torchvision.utils import _log_api_usage_once
from mydatasets.feature_acoustic import extract_melspectrogram_from_waveform
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from models.AST_Model import ASTModel
from main.opts import ARGS
from torchvision.models import efficientnet, EfficientNet_B2_Weights
from leaf_pytorch.frontend import Leaf
import librosa


# module in resnet
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


# module in resnet
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# Modified from torchvision.models.ResNet
class ResNetWrapper(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init

        self.reset_parameters(zero_init_residual)

    def reset_parameters(self, zero_init_residual: bool = False):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def trainable_parameters(self):
        pass

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


# pretrained resnet101
class EmoNet_ResNet101(nn.Module):

    def __init__(self, num_classes):
        super(EmoNet_ResNet101, self).__init__()
        self.resnet101 = ResNetWrapper(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)

        self.reset_parameters()
        # self.rnn = nn.LSTM()
        # self.classifier = nn.Linear(512, num_classes)

    def reset_parameters(self):
        pretrained_resnet101_state_dict = resnet101(weights=ResNet101_Weights).state_dict()
        resnet101_state_dict = self.resnet101.state_dict()

        for (m, pretrained_m) in zip(resnet101_state_dict.keys(), pretrained_resnet101_state_dict.keys()):

            if m == pretrained_m and not m.startswith("fc"):
                resnet101_state_dict[m] = pretrained_resnet101_state_dict[pretrained_m]

    def trainable_parameters(self):
        return list(self.resnet101.parameters())

    def forward(self, x, length=None):
        with torch.no_grad():
            mel_spec = self._compute_Mel_Spectrogram_feature_v2(x)
        logits = self.resnet101(mel_spec)
        return logits

    def _compute_Mel_Spectrogram_feature_v2(self, x):
        """
        :param x: [B,waveform]
        :return: Mel_Spectrogram:[B,128,L]
        """
        to_batch_list = []
        with torch.no_grad():
            for i in range(x.shape[0]):
                mel_spec = _waveform_to_3d_mel_spec(x[i].cpu().numpy(), sample_rate=16000)
                to_batch_list.append(torch.as_tensor(mel_spec).unsqueeze(0).cpu().numpy())
        mel_spec_batch = np.concatenate(to_batch_list, axis=0)
        return torch.tensor(mel_spec_batch, dtype=x.dtype, device=x.device)


# pretrained resnet34
class EmoNet_ResNet34(nn.Module):

    def __init__(self, num_classes):
        super(EmoNet_ResNet34, self).__init__()
        self.resnet34 = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
        self.reset_parameters()

    def forward(self, x, length=None):
        with torch.no_grad():
            mel_spec = self._compute_Mel_Spectrogram_feature_v2(x)
        logits = self.resnet34(mel_spec)
        return logits

    def reset_parameters(self):
        pretrained_resnet34_state_dict = resnet34(weights=ResNet34_Weights).state_dict()
        resnet34_state_dict = self.resnet34.state_dict()
        for (m, pretrained_m) in zip(resnet34_state_dict.keys(), pretrained_resnet34_state_dict.keys()):

            if m == pretrained_m and not m.startswith("fc"):
                resnet34_state_dict[m] = pretrained_resnet34_state_dict[pretrained_m]

    def trainable_parameters(self):
        return list(self.resnet34.parameters())

    def _compute_Mel_Spectrogram_feature_v2(self, x):
        """
        :param x: [B,waveform]
        :return: Mel_Spectrogram:[B,128,L]
        """
        to_batch_list = []
        with torch.no_grad():
            for i in range(x.shape[0]):
                mel_spec = _waveform_to_3d_mel_spec(x[i].cpu().numpy(), sample_rate=16000)
                to_batch_list.append(torch.as_tensor(mel_spec).unsqueeze(0).cpu().numpy())
        mel_spec_batch = np.concatenate(to_batch_list, axis=0)
        return torch.tensor(mel_spec_batch, dtype=x.dtype, device=x.device)


# pretrained resnet18
class EmoNet_ResNet18(nn.Module):

    def __init__(self, num_classes):
        super(EmoNet_ResNet18, self).__init__()
        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.reset_parameters()

    def forward(self, x, length=None):
        with torch.no_grad():
            mel_spec = self._compute_Mel_Spectrogram_feature_v2(x)
        logits = self.resnet18(mel_spec)
        return logits

    def reset_parameters(self):
        pretrained_resnet18_state_dict = resnet18(weights=ResNet18_Weights).state_dict()
        resnet18_state_dict = self.resnet18.state_dict()

        for (m, pretrained_m) in zip(resnet18_state_dict.keys(), pretrained_resnet18_state_dict.keys()):

            if m == pretrained_m and not m.startswith("fc"):
                resnet18_state_dict[m] = pretrained_resnet18_state_dict[pretrained_m]

    def trainable_parameters(self):
        return list(self.resnet18.parameters())

    def _compute_Mel_Spectrogram_feature_v2(self, x):
        """
        :param x: [B,waveform]
        :return: Mel_Spectrogram:[B,128,L]
        """
        to_batch_list = []
        with torch.no_grad():
            for i in range(x.shape[0]):
                mel_spec = _waveform_to_3d_mel_spec(x[i].cpu().numpy(), sample_rate=16000)
                to_batch_list.append(torch.as_tensor(mel_spec).unsqueeze(0).cpu().numpy())
        mel_spec_batch = np.concatenate(to_batch_list, axis=0)
        return torch.tensor(mel_spec_batch, dtype=x.dtype, device=x.device)


# module for AlexNet in paper
def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


# module for AlexNet in paper
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


# AlexNet used in paper
'''
2 Models Available:
   - SER_AlexNet     : AlexNet model from pyTorch (CNN features layer + FC classifier layer)
'''


class SER_AlexNet(nn.Module):
    """
    Reference:
    https://pytorch.org/docs/stable/torchvision/models.html#id1

    AlexNet model from torchvision package. The model architecture is slightly
    different from the original model.
    See: AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.


    Parameters
    ----------
    num_classes : int
    in_ch   : int
        The number of input channel.
        Default AlexNet input channels is 3. Set this parameters for different
            numbers of input channels.
    pretrained  : bool
        To initialize the weight of AlexNet.
        Set to 'True' for AlexNet pre-trained weights.

    Input
    -----
    Input dimension (N,C,H,W)

    N   : batch size
    C   : channels
    H   : Height
    W   : Width

    Output
    ------
    logits (before Softmax)

    """

    def __init__(self, num_classes=4, in_ch=1, pretrained=True):
        super(SER_AlexNet, self).__init__()

        model = torchvision.models.alexnet(pretrained=pretrained)
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier

        if in_ch != 3:
            self.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
            init_layer(self.features[0])

        self.classifier[6] = nn.Linear(4096, num_classes)

        self._init_weights(pretrained=pretrained)

        print('\n<< SER AlexNet Finetuning model initialized >>\n')

    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        x_ = torch.flatten(x, 1)
        out = self.classifier(x_)

        return x, out

    def _init_weights(self, pretrained=True):

        init_layer(self.classifier[6])

        if pretrained == False:
            init_layer(self.features[0])
            init_layer(self.features[3])
            init_layer(self.features[6])
            init_layer(self.features[8])
            init_layer(self.features[10])
            init_layer(self.classifier[1])
            init_layer(self.classifier[4])


# pretrained AlexNet
class EmoNet_AlexNet(nn.Module):

    def __init__(self, num_classes):
        super(EmoNet_AlexNet, self).__init__()
        self.alexnet_model = SER_AlexNet(num_classes=num_classes, in_ch=1, pretrained=True)
        # self.post_spec_dropout = nn.Dropout(p=0.1)
        # self.post_spec_layer = nn.Linear(9216, 128)  # 9216 for cnn, 32768 for ltsm s, 65536 for lstm l
        self.feature_extractor = Leaf(n_filters=128)

    def forward(self, x, length=None):
        # audio_spec: [batch, 3, 256, 384] -> [batch, 3, 224, 526]
        """
        with torch.no_grad():
            audio_spec = self._compute_Mel_Spectrogram_feature(x)
        """

        x = x.unsqueeze(1)
        acoustic_feature = self.feature_extractor(x)
        acoustic_feature = acoustic_feature.unsqueeze(1)
        """
        with torch.no_grad():
            audio_spec = self._compute_Mel_Spectrogram_feature_v2(x)
        """
        # spectrogram - SER_CNN
        # audio_spec, output_spec_t = self.alexnet_model(audio_spec)  # [batch, 256, 6, 6], []
        audio_spec, output_spec_t = self.alexnet_model(acoustic_feature)  # [batch, 256, 6, 6], []
        # audio_spec = audio_spec.reshape(audio_spec.shape[0], audio_spec.shape[1], -1)  # [batch, 256, 36]
        # audio_spec_ = torch.flatten(audio_spec, 1)  # [batch, 9216]
        # audio_spec_d = self.post_spec_dropout(audio_spec_)  # [batch, 9216]
        # audio_spec_p = F.relu(self.post_spec_layer(audio_spec_d), inplace=False)  # [batch, 128]
        return output_spec_t

    def trainable_parameters(self):
        return list(self.alexnet_model.parameters())+list(self.feature_extractor.parameters())

    def _compute_Mel_Spectrogram_feature(self, x):
        """
        :param x: [B,waveform]
        :return: Mel_Spectrogram:[B,128,L]
        """
        with torch.no_grad():
            waveform = x[0]
            mel_spec = extract_melspectrogram_from_waveform(waveform.cpu().numpy(), sr=16000, n_fft=640,
                                                            hop_length=160, fmax=8000, n_mels=200)
            mel_spec = torch.as_tensor(mel_spec, dtype=x.dtype, device=torch.device('cpu')).unsqueeze(0).unsqueeze(
                0).numpy()
            mel_spec = _spec_to_rgb(mel_spec)
            mel_spec_ = torch.as_tensor(mel_spec, dtype=x.dtype, device=torch.device('cpu')).unsqueeze(0)
            for i in range(x.shape[0] - 1):
                mel_spec = extract_melspectrogram_from_waveform(waveform.cpu().numpy(), sr=16000, n_fft=640,
                                                                hop_length=160, fmax=8000, n_mels=200)
                mel_spec = torch.as_tensor(mel_spec, dtype=x.dtype, device=torch.device('cpu')).unsqueeze(0).unsqueeze(
                    0).numpy()
                mel_spec = _spec_to_rgb(mel_spec)
                mel_spec_ = np.concatenate(
                    [mel_spec_, torch.as_tensor(mel_spec, dtype=x.dtype, device=torch.device('cpu')).unsqueeze(0)],
                    axis=0)
        return torch.tensor(mel_spec_, dtype=x.dtype, device=x.device)

    def _compute_Mel_Spectrogram_feature_v2(self, x):
        """
        :param x: [B,waveform]
        :return: Mel_Spectrogram:[B,128,L]
        """
        to_batch_list = []
        with torch.no_grad():
            for i in range(x.shape[0]):
                mel_spec = _waveform_to_3d_mel_spec(x[i].cpu().numpy(), sample_rate=16000)
                to_batch_list.append(torch.as_tensor(mel_spec).unsqueeze(0).cpu().numpy())
        mel_spec_batch = np.concatenate(to_batch_list, axis=0)
        return torch.tensor(mel_spec_batch, dtype=x.dtype, device=x.device)


# pretrained EfficientNet
class EmoNet_EfficientNet(nn.Module):

    def __init__(self, num_classes):
        super(EmoNet_EfficientNet, self).__init__()
        self.efficientNet = efficientnet.efficientnet_b2(num_classes=num_classes)
        self.reset_parameters()

    def forward(self, x, length=None):
        with torch.no_grad():
            audio_spec = self._compute_Mel_Spectrogram_feature_v2(x)
        logits = self.efficientNet(audio_spec)
        return logits

    def trainable_parameters(self):
        return self.parameters()

    def reset_parameters(self):
        pretrained_eff_b2_state_dict = efficientnet.efficientnet_b2(weights=EfficientNet_B2_Weights).state_dict()
        eff_b2_state_dict = self.efficientNet.state_dict()

        for (m, pretrained_m) in zip(eff_b2_state_dict.keys(), pretrained_eff_b2_state_dict.keys()):

            if m == pretrained_m and not m.startswith("fc"):
                eff_b2_state_dict[m] = pretrained_eff_b2_state_dict[pretrained_m]

    def _compute_Mel_Spectrogram_feature_v2(self, x):
        """
        :param x: [B,waveform]
        :return: Mel_Spectrogram:[B,128,L]
        """
        to_batch_list = []
        with torch.no_grad():
            for i in range(x.shape[0]):
                mel_spec = _waveform_to_3d_mel_spec(x[i].cpu().numpy(), sample_rate=16000)
                to_batch_list.append(torch.as_tensor(mel_spec).unsqueeze(0).cpu().numpy())
        mel_spec_batch = np.concatenate(to_batch_list, axis=0)
        return torch.tensor(mel_spec_batch, dtype=x.dtype, device=x.device)


# pretrained SSAST
class EmoNet_SSAST(nn.Module):

    def __init__(self, num_classes, time_length=7):
        super(EmoNet_SSAST, self).__init__()
        self.pretrained_model_name = ARGS.MODEL_NAME_SSAST
        self.input_tdim = (time_length * 100)
        self.ast_model = ASTModel(label_dim=num_classes,
                                  fshape=128, tshape=2, fstride=128, tstride=1,
                                  input_fdim=128, input_tdim=self.input_tdim, model_size='tiny',
                                  pretrain_stage=False, load_pretrained_mdl_path=self.pretrained_model_name)
        self.learnable_frontend = Leaf(n_filters=128)
    def forward(self, x,length=None):
        # [10, input_tdim, 128]
        x = x.unsqueeze(1)
        x = self.learnable_frontend(x)
        x = x.permute([0,2,1])
        """
        with torch.no_grad():
            x = self._compute_Mel_Spectrogram_1D_feature(x)
        """
        logits = self.ast_model(x, task='ft_avgtok')
        return logits

    def trainable_parameters(self):
        return list(self.ast_model.parameters())+list(self.learnable_frontend.parameters())

    def reset_parameters(self):
        pass

    def _compute_Mel_Spectrogram_1D_feature(self, x):
        """
        :param x: [B,waveform]
        :return: Mel_Spectrogram:[B,128,L]
        """
        to_batch_list = []
        with torch.no_grad():
            for i in range(x.shape[0]):
                mel_spec = librosa.feature.melspectrogram(y=x[i].cpu().numpy(), sr=16000, hop_length=160,
                                                          win_length=400, n_mels=128)
                log_mel_spec = np.log(mel_spec.T.astype(np.float32) + 1e-10)
                to_batch_list.append(torch.as_tensor(log_mel_spec).unsqueeze(0).cpu().numpy())
        mel_spec_batch = np.concatenate(to_batch_list, axis=0)
        return torch.tensor(mel_spec_batch, dtype=x.dtype, device=x.device)


# SER Toolkits

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


def _waveform_to_3d_mel_spec(waveform, sample_rate):
    _, spec = log_specgram(waveform, sample_rate)

    spec_3d = get_3d_spec(spec)

    trans_spec = np.transpose(spec_3d, (2, 0, 1))

    return trans_spec


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
    right1 = np.concatenate([Sxx_in[:, 0].reshape((h, -1)), Sxx_in], axis=1)[:, :-1]
    delta = (Sxx_in - right1)[:, 1:]
    delta_pad = delta[:, 0].reshape((h, -1))
    delta = np.concatenate([delta_pad, delta], axis=1)
    right2 = np.concatenate([delta[:, 0].reshape((h, -1)), delta], axis=1)[:, :-1]
    delta2 = (delta - right2)[:, 1:]
    delta2_pad = delta2[:, 0].reshape((h, -1))
    delta2 = np.concatenate([delta2_pad, delta2], axis=1)
    base = (Sxx_in - base_mean) / base_std
    delta = (delta - delta_mean) / delta_std
    delta2 = (delta2 - delta2_mean) / delta2_std
    stacked = [arr.reshape((h, w, 1)) for arr in (base, delta, delta2)]
    return np.concatenate(stacked, axis=2)


if __name__ == '__main__':
    # x = torch.randn(4, 3, 224, 224)
    waveform = torch.randn(4, 112000)

    net = EmoNet_SSAST(num_classes=4)

    # waveform = waveform.cuda()
    # net.cuda()

    out = net(waveform)
    print(
        net(waveform).shape
    )

