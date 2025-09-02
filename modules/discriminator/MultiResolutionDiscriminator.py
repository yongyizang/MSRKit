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

class ResolutionDiscriminatorBlock(nn.Module):
    def __init__(
        self, 
        window_length: int, 
        nch: int = 1, 
        sample_rate: int = 48000, 
        hop_factor: float = 0.25, 
        bands: tp.List[tp.Tuple[float, float]] = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)], 
        norm: bool = True, 
        hidden_channels: int = 32
    ):
        super().__init__()
        self.window_length = window_length
        self.hop_length = int(window_length * hop_factor)
        self.sample_rate = sample_rate
        self.nch = nch

        n_fft_bins = window_length // 2 + 1
        self.bands = [(int(b[0] * n_fft_bins), int(b[1] * n_fft_bins)) for b in bands]

        self.band_discriminators = nn.ModuleList()
        for _ in self.bands:
            layers = nn.ModuleList([
                DiscriminatorBlock2d(2 * nch, hidden_channels, (3, 9), (1, 1), padding=(1, 4), norm=norm),
                DiscriminatorBlock2d(hidden_channels, hidden_channels, (3, 9), (1, 2), padding=(1, 4), norm=norm),
                DiscriminatorBlock2d(hidden_channels, hidden_channels, (3, 9), (1, 2), padding=(1, 4), norm=norm),
                DiscriminatorBlock2d(hidden_channels, hidden_channels, (3, 9), (1, 2), padding=(1, 4), norm=norm),
                DiscriminatorBlock2d(hidden_channels, hidden_channels, (3, 3), (1, 1), padding=(1, 1), norm=norm),
            ])
            self.band_discriminators.append(layers)

        self.output_conv = DiscriminatorBlock2d(hidden_channels, 1, (3, 3), (1, 1), padding=(1, 1), norm=norm, activation=False)

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        fmaps = []
        band_outputs = []
        
        x_spec = torch.stft(
            x.reshape(-1, x.shape[-1]),
            n_fft=self.window_length,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=torch.hann_window(self.window_length, device=x.device),
            return_complex=True
        )
        
        x_ri = torch.stack([x_spec.real, x_spec.imag], dim=1)
        
        B, C, _ = x.shape
        x_ri = x_ri.view(B, C * 2, x_ri.shape[-2], x_ri.shape[-1])
        x_ri = rearrange(x_ri, 'b c f t -> b c t f')

        for i, (band_start, band_end) in enumerate(self.bands):
            x_band = x_ri[..., band_start:band_end]
            
            disc_stack = self.band_discriminators[i]
            for layer in disc_stack:
                x_band = layer(x_band)
                fmaps.append(x_band)
            band_outputs.append(x_band)

        x_combined = torch.cat(band_outputs, dim=-1)
        score = self.output_conv(x_combined)
        
        return score, fmaps

class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, nch: int = 1, sample_rate: int = 48000, window_lengths: tp.List[int] = [2048, 1024, 512], hop_factor: float = 0.25, bands: tp.List[tp.Tuple[float, float]] = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)], norm: bool = True, hidden_channels: int = 32):
        super().__init__()
        self.nch = nch
        self.sample_rate = sample_rate
        self.window_lengths = window_lengths
        self.hop_factor = hop_factor
        self.bands = bands
        self.norm = norm
        self.hidden_channels = hidden_channels
        self.discriminators = nn.ModuleList([ResolutionDiscriminatorBlock(window_length, nch, sample_rate, hop_factor, bands, norm, hidden_channels) for window_length in window_lengths])
    
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

    model = MultiResolutionDiscriminator(
        nch=N_CHANNELS,
        sample_rate=SAMPLE_RATE,
        window_lengths=[2048, 1024, 512],
        hop_factor=0.25,
        bands=[(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)],
        norm=True,
        hidden_channels=32
    )

    dummy_audio = torch.randn(2, N_CHANNELS, SAMPLE_RATE)
    
    scores_list, fmaps_list = model(dummy_audio)

    for score, fmap in zip(scores_list, fmaps_list):
        for fm in fmap:
            print(fm.shape)