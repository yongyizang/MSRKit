import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import typing as tp

class DiscriminatorBlock1d(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int, 
        padding: int, 
        groups: int = 1, 
        norm: bool = True, 
        activation: bool = True
    ):
        super().__init__()
        self.activation = activation
        conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        
        if norm:
            self.conv = nn.utils.spectral_norm(conv_layer)
        else:
            self.conv = conv_layer
        
        if self.activation:
            self.act_fn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.activation:
            x = self.act_fn(x)
        return x

class DiscriminatorBlock2d(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: tp.Tuple[int, int], 
        stride: tp.Tuple[int, int], 
        padding: tp.Tuple[int, int], 
        norm: bool = True, 
        activation: bool = True
    ):
        super().__init__()
        self.activation = activation
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        if norm:
            self.conv = nn.utils.spectral_norm(conv_layer)
        else:
            self.conv = conv_layer
        
        if self.activation:
            self.act_fn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.activation:
            x = self.act_fn(x)
        return x

class PeriodDiscriminatorBlock(nn.Module):
    def __init__(self, period: int, nch: int = 1, norm: bool = True):
        super().__init__()
        self.period = period
        self.nch = nch
        
        channels = [
            (nch, 32), (32, 128), (128, 512), (512, 1024), (1024, 1024)
        ]
        
        self.layers = nn.ModuleList()
        for in_ch, out_ch in channels:
            self.layers.append(
                DiscriminatorBlock2d(in_ch, out_ch, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0), norm=norm)
            )
            
        self.output_conv = DiscriminatorBlock2d(1024, 1, (3, 1), (1, 1), padding=(1, 0), norm=norm, activation=False)

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        fmaps = []
        
        b, c, t = x.shape
        if t % self.period != 0:
            pad_len = self.period - (t % self.period)
            x = F.pad(x, (0, pad_len), "reflect")
        
        x = rearrange(x, 'b c (l p) -> b c l p', p=self.period)

        for layer in self.layers:
            x = layer(x)
            fmaps.append(x)
        
        score = self.output_conv(x)
        
        return score, fmaps

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, nch: int = 1, sample_rate: int = 48000, periods: tp.List[int] = [2, 3, 5, 7, 11], norm: bool = True):
        super().__init__()
        self.nch = nch
        self.sample_rate = sample_rate
        self.periods = periods
        self.norm = norm
        self.discriminators = nn.ModuleList([PeriodDiscriminatorBlock(period, nch, norm) for period in periods])

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        scores = []
        fmaps = []
        for discriminator in self.discriminators:
            score, fmap = discriminator(x)
            scores.append(score)
            fmaps.append(fmap)
        return scores, fmaps

if __name__ == '__main__':
    N_CHANNELS = 1
    SAMPLE_RATE = 48000

    model = MultiPeriodDiscriminator(
        nch=N_CHANNELS,
        sample_rate=SAMPLE_RATE,
        periods=[2, 3, 5, 7, 11],
        norm=True
    )

    dummy_audio = torch.randn(2, N_CHANNELS, SAMPLE_RATE)
    
    scores_list, fmaps_list = model(dummy_audio)

    for score, fmap in zip(scores_list, fmaps_list):
        for fm in fmap:
            print(fm.shape)