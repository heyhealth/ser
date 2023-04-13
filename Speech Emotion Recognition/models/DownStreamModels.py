from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):

    def __init__(self, in_dim, mid_dim, num_classes):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mid_dim, num_classes),
            nn.ReLU(),
            nn.Dropout(0.5),

        )

    def forward(self, x):
        logits = self.mlp(x)
        return logits


"""EXPLORING WAV2VEC 2.0 FINE TUNING FOR IMPROVED SPEECH EMOTION RECOGNITION"""


class LinearHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        # Input: (B, L, C)
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.Tanh()
        self.l2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = self.act(self.l1(x))
        x = self.l2(x)
        return x

    def reset_parameters(self):
        self.l1.reset_parameters()
        self.l2.reset_parameters()


"""EXPLORING WAV2VEC 2.0 FINE TUNING FOR IMPROVED SPEECH EMOTION RECOGNITION"""


class RNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        # Input: (B, L, C)
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = F.relu(x.mean(1))
        return x
