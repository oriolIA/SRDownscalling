"""
ESRGAN-like Super Resolution model for WRF data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block for ESRGAN."""

    def __init__(self, in_channels: int = 64, growth_channels: int = 32, num_layers: int = 5):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Conv2d(in_channels + i * growth_channels, growth_channels, 3, padding=1)
            )

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv_out = nn.Conv2d(in_channels + num_layers * growth_channels, 64, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = [x]
        for layer in self.layers:
            out = layer(torch.cat(inputs, dim=1))
            out = self.lrelu(out)
            inputs.append(out)
        return self.conv_out(torch.cat(inputs, dim=1)) + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block (ESRGAN)."""

    def __init__(self, in_channels: int = 64, growth_channels: int = 32):
        super().__init__()
        self.rdbs = nn.ModuleList([
            ResidualDenseBlock(in_channels, growth_channels) for _ in range(3)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for rdb in self.rdbs:
            out = rdb(out)
        return out * 0.2 + x  # Residual scaling


class ESRGAN(nn.Module):
    """ESRGAN-like Super Resolution Network.

    Args:
        in_channels: Number of input channels (meteorological variables)
        out_channels: Number of output channels
        num_rrdb: Number of RRDB blocks
        scale_factor: Upscaling factor (2, 4, 8)
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 2,
        num_rrdb: int = 12,
        scale_factor: int = 2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        # Feature extraction
        self.conv_first = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        # RRDB trunk
        self.rrdb_trunk = nn.Sequential(
            *[RRDB() for _ in range(num_rrdb)]
        )

        # Upsampling
        self.upscale = nn.ModuleList()
        for _ in range(int(scale_factor / 2)):
            self.upscale.extend([
                nn.Conv2d(64, 256, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.PixelShuffle(2),
            ])

        # Final output
        self.conv_last = nn.Conv2d(64, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial feature extraction
        feat = self.lrelu(self.conv_first(x))

        # RRDB trunk
        feat = self.rrdb_trunk(feat)

        # Upsample
        for layer in self.upscale:
            feat = layer(feat)

        # Output
        return self.conv_last(feat)
