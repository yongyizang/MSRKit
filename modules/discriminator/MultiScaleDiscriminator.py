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

class ScaleDiscriminatorBlock(nn.Module):
    def __init__(self, sample_rate: int, downsample_rate: int, nch: int = 1, norm: bool = True):
        super().__init__()
        self.sample_rate = sample_rate
        self.downsample_rate = downsample_rate
        self.nch = nch

        layer_params = [
            {'in': nch, 'out': 16, 'k': 15, 's': 1, 'g': 1, 'p': 7},
            {'in': 16, 'out': 64, 'k': 41, 's': 4, 'g': 4, 'p': 20},
            {'in': 64, 'out': 256, 'k': 41, 's': 4, 'g': 16, 'p': 20},
            {'in': 256, 'out': 1024, 'k': 41, 's': 4, 'g': 64, 'p': 20},
            {'in': 1024, 'out': 1024, 'k': 41, 's': 4, 'g': 256, 'p': 20},
            {'in': 1024, 'out': 1024, 'k': 5, 's': 1, 'g': 1, 'p': 2},
        ]

        self.layers = nn.ModuleList()
        for p in layer_params:
            self.layers.append(
                DiscriminatorBlock1d(p['in'], p['out'], p['k'], p['s'], p['p'], p['g'], norm=norm)
            )

        self.output_conv = DiscriminatorBlock1d(1024, 1, 3, 1, 1, norm=norm, activation=False)

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        fmaps = []
        
        if self.downsample_rate > 1:
            x = F.interpolate(x, scale_factor=1./self.downsample_rate, mode='linear', align_corners=False)
        
        for layer in self.layers:
            x = layer(x)
            fmaps.append(x)
            
        score = self.output_conv(x)
        
        return score, fmaps

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, sample_rate: int, downsample_rates: tp.List[int] = [2, 4], nch: int = 1, norm: bool = True):
        super().__init__()
        self.sample_rate = sample_rate
        self.downsample_rates = downsample_rates
        self.nch = nch
        self.norm = norm
        self.discriminators = nn.ModuleList([ScaleDiscriminatorBlock(sample_rate, downsample_rate, nch, norm) for downsample_rate in downsample_rates])
    
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

    model = MultiScaleDiscriminator(
        nch=N_CHANNELS,
        sample_rate=SAMPLE_RATE,
        downsample_rates=[2, 4],
        norm=True
    )

    dummy_audio = torch.randn(2, N_CHANNELS, SAMPLE_RATE)
    
    scores_list, fmaps_list = model(dummy_audio)

    for score, fmap in zip(scores_list, fmaps_list):
        for fm in fmap:
            print(fm.shape)