"""
Simple UNet-based Super Resolution model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetSR(nn.Module):
    """UNet for Super Resolution of WRF data.

    Downscales from coarse (9km) to fine (100m) resolution.
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 2,
        n_filters: int = 64,
        scale_factor: int = 2
    ):
        super().__init__()
        self.n_filters = n_filters
        self.scale_factor = scale_factor

        # Initial upsampling to match expected resolution
        self.init_upsample = nn.Sequential(
            nn.Conv2d(in_channels, n_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
        )

        # Encoder
        self.enc1 = DoubleConv(n_filters, n_filters)
        self.enc2 = DoubleConv(n_filters, n_filters * 2)
        self.enc3 = DoubleConv(n_filters * 2, n_filters * 4)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(n_filters * 4, n_filters * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(n_filters * 8, n_filters * 4, 2, stride=2)
        self.dec3 = DoubleConv(n_filters * 8, n_filters * 4)

        self.up2 = nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 2, stride=2)
        self.dec2 = DoubleConv(n_filters * 4, n_filters * 2)

        self.up1 = nn.ConvTranspose2d(n_filters * 2, n_filters, 2, stride=2)
        self.dec1 = DoubleConv(n_filters * 2, n_filters)

        # Output
        self.final = nn.Conv2d(n_filters, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial upsampling
        x = self.init_upsample(x)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder
        d3 = self.up3(b)
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)


if __name__ == "__main__":
    # Test
    model = UNetSR(in_channels=7, out_channels=2, scale_factor=2)
    x = torch.randn(1, 7, 50, 51)  # d01 shape
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
