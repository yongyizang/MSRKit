import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x * x, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed

class ConvNeXt2DBlock(nn.Module):
    def __init__(
        self,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        input_dim: int,
        output_dim: int,
        mode: str = 'normal',
    ):
        super().__init__()
        self.mode = mode
        assert mode in ['normal', 'transposed'], 'mode must be either "normal" or "transposed"'
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        dim = input_dim
        if mode == 'normal':
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=self.padding)
            self.residual_conv = nn.Conv2d(dim, output_dim, kernel_size=kernel_size, stride=stride, padding=self.padding)
        else:
            self.dwconv = nn.ConvTranspose2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=self.padding)
            self.residual_conv = nn.ConvTranspose2d(dim, output_dim, kernel_size=kernel_size, stride=stride, padding=self.padding)
        self.norm = RMSNorm(dim)
        self.n_hidden = int(4 * dim / 3)
        self.pwconv1 = nn.Linear(dim, self.n_hidden * 2)
        self.pwconv2 = nn.Linear(self.n_hidden, output_dim)

    def forward(self, x: torch.Tensor, original_shape: Optional[tuple[int, int]] = None) -> torch.Tensor:
        # If in transposed mode and original_shape is provided, adjust output size
        if self.mode == 'transposed' and original_shape is not None:
            # Calculate output size for ConvTranspose2d
            output_size = (x.shape[0], self.residual_conv.out_channels, original_shape[0], original_shape[1])
            residual = self.residual_conv(x, output_size=output_size)
            x = self.dwconv(x, output_size=output_size)
        else:
            residual = self.residual_conv(x)
            x = self.dwconv(x)
        x = x.transpose(1, 3)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = F.silu(x[:, :, :, :self.n_hidden]) * x[:, :, :, self.n_hidden:]
        x = self.pwconv2(x)
        x = x.transpose(1, 3)
        x = residual + x
        return x

if __name__ == "__main__":
    model = ConvNeXt2DBlock(kernel_size=(7, 7), stride=(1, 1), input_dim=512, output_dim=512)
    input = torch.randn(4, 512, 100, 100) # (B, C, H, W)
    output = model(input)
    print(output.shape)

    # test normal and 
    model = ConvNeXt2DBlock(kernel_size=(7, 7), stride=(2, 2), input_dim=512, output_dim=512, mode='normal')
    input = torch.randn(4, 512, 100, 100) # (B, C, H, W)
    output = model(input)
    print(output.shape)

    # test transposed mode
    model = ConvNeXt2DBlock(kernel_size=(7, 7), stride=(2, 2), input_dim=512, output_dim=512, mode='transposed')
    input = torch.randn(4, 512, 100, 100) # (B, C, H, W)
    output = model(input)
    print(output.shape)