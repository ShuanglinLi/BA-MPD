import torch.nn.functional as F
import torchaudio
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=(2, 2)):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool_size)


class PannsCnn(nn.Module):
    """PANNs-style CNN14 classifier for single-label BEANS-CBI."""

    def __init__(self, n_classes, n_mels=64, sample_rate=32000):
        super().__init__()
        channels = [64, 128, 256, 512, 1024, 2048]
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=320,
            n_mels=n_mels,
            f_min=50,
            f_max=14000,
            power=2.0,
        )
        self.bn0 = nn.BatchNorm2d(1)
        blocks = []
        in_channels = 1
        for out_channels in channels:
            blocks.append(ConvBlock(in_channels, out_channels))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.fc1 = nn.Linear(channels[-1], 2048)
        self.fc_out = nn.Linear(2048, n_classes)
        self.dropout = nn.Dropout(0.2)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, waveform):
        x = waveform.mean(dim=1) if waveform.dim() == 3 else waveform
        x = self.mel(x).clamp_min(1e-10).log().unsqueeze(1)
        x = self.bn0(x)
        for block in self.blocks:
            x = block(x)
            x = F.dropout(x, p=0.2, training=self.training)
        x = x.mean(dim=2)
        x = x.max(dim=2).values + x.mean(dim=2)
        x = F.relu_(self.fc1(x))
        x = self.dropout(x)
        return self.fc_out(x)

