import torch
import torch.nn as nn
from einops import rearrange
import typing as tp

def get_2d_padding(kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int] = (1, 1)) -> tp.Tuple[int, int]:
    return (((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2)

class FrequencyDiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, norm=True, dilation=(1, 1)):
        super().__init__()
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        if norm:
            self.conv = nn.utils.spectral_norm(conv_layer)
        else:
            self.conv = conv_layer
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x))


class FrequencyDiscriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels=8, norm=True):
        super().__init__()
        
        channels = [
            (in_channels, hidden_channels),
            (hidden_channels, hidden_channels * 2),
            (hidden_channels * 2, hidden_channels * 4),
            (hidden_channels * 4, hidden_channels * 8),
            (hidden_channels * 8, hidden_channels * 16),
        ]
        dilations = [1, 2, 4]
        kernel_rect = (3, 9)
        kernel_sq = (3, 3)
        stride_downsample = (1, 2)
        stride_none = (1, 1)

        self.layers = nn.ModuleList()

        self.layers.append(
            FrequencyDiscriminatorBlock(
                channels[0][0], channels[0][1], kernel_rect, stride_none,
                padding=get_2d_padding(kernel_rect), norm=norm
            )
        )

        for i, dilation in enumerate(dilations):
            in_ch, out_ch = channels[i+1]
            self.layers.append(
                FrequencyDiscriminatorBlock(
                    in_ch, out_ch, kernel_rect, stride_downsample,
                    padding=get_2d_padding(kernel_rect, (dilation, 1)),
                    norm=norm, dilation=(dilation, 1)
                )
            )

        in_ch, out_ch = channels[-1]
        self.layers.append(
            FrequencyDiscriminatorBlock(
                in_ch, out_ch, kernel_sq, stride_none,
                padding=get_2d_padding(kernel_sq), norm=norm
            )
        )

        self.layers.append(
             FrequencyDiscriminatorBlock(
                out_ch, hidden_channels * 32, kernel_sq, stride_none,
                padding=get_2d_padding(kernel_sq), norm=norm
            )
        )

        self.output_conv = nn.Conv2d(hidden_channels * 32, 1, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        fmaps = []
        x = rearrange(x, 'b c f t -> b c t f')
        
        for layer in self.layers:
            x = layer(x)
            fmaps.append(x)
        
        score = self.output_conv(x)
        return score, fmaps


class MultiFrequencyDiscriminator(nn.Module):
    def __init__(self, nch, window_sizes, hidden_channels=8, sample_rate=48000, norm=True):
        super().__init__()
        self.nch = nch
        self.window_sizes = window_sizes
        self.sample_rate = sample_rate
        self.eps = torch.finfo(torch.float32).eps

        self.discriminators = nn.ModuleList(
            [FrequencyDiscriminator(2 * nch, hidden_channels, norm=norm) for _ in self.window_sizes]
        )

    def forward(self, est):
        B, nch, _ = est.shape
        assert nch == self.nch, f"Input channels {nch} do not match model's expected {self.nch}"

        norm_factor = (est.pow(2).sum((1, 2), keepdim=True) + self.eps).sqrt()
        est = est / norm_factor
        est = est.view(-1, est.shape[-1])

        scores = []
        fmaps = []

        for i, disc in enumerate(self.discriminators):
            win_length = self.window_sizes[i]
            hop_length = win_length // 2
            
            est_spec = torch.stft(
                est.float(),
                n_fft=win_length,
                hop_length=hop_length,
                win_length=win_length,
                window=torch.hann_window(win_length, device=est.device),
                return_complex=True
            )
            
            est_ri = torch.stack([est_spec.real, est_spec.imag], dim=1)
            est_ri = est_ri.view(B, nch * 2, est_ri.shape[-2], est_ri.shape[-1]).type(est.type())

            valid_freq_bins = int(est_ri.shape[2] * self.sample_rate / 48000)
            est_ri_sliced = est_ri[:, :, :valid_freq_bins, :].contiguous()
            
            score, fmap_group = disc(est_ri_sliced)
            scores.append(score)
            fmaps.append(fmap_group)

        return scores, fmaps

if __name__ == '__main__':
    N_CHANNELS = 1
    HIDDEN_CHANNELS = 8
    SAMPLE_RATE = 48000
    WINDOW_SIZES = [2048, 1024, 512]
    
    model = MultiFrequencyDiscriminator(
        nch=N_CHANNELS,
        window_sizes=WINDOW_SIZES,
        hidden_channels=HIDDEN_CHANNELS,
        sample_rate=SAMPLE_RATE,
        norm=True
    )
    
    dummy_audio = torch.randn(2, N_CHANNELS, SAMPLE_RATE)
    
    scores_list, fmaps_list = model(dummy_audio)

    for i, (score, fmap) in enumerate(zip(scores_list, fmaps_list)):
        for j, fm in enumerate(fmap):
            print(fm.shape)