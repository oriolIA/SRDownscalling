"""
SRGAN - Super-Resolution Generative Adversarial Network

Complete architecture combining:
- Generator: RRDB + Attention + Upsampling
- Discriminator: PatchGAN with Spectral Normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .blocks import RRDB, ChannelAttention


class Generator(nn.Module):
    """SRGAN Generator with RRDB blocks and attention."""
    
    def __init__(
        self,
        in_channels: int = 7,  # U, V, W, T, P, HGT, TKE
        num_rrdb: int = 16,
        nf: int = 64,
        gc: int = 32,
        scale: int = 2,
    ):
        super().__init__()
        self.scale = scale
        
        # Feature extraction
        self.conv_first = nn.Conv2d(in_channels, nf, 3, 1, 1)
        
        # RRDB trunk
        self.rrdb_trunk = nn.Sequential(
            *[RRDB(nf, gc) for _ in range(num_rrdb)]
        )
        
        # Attention before upsampling
        self.ca = ChannelAttention(nf)
        
        # Reconstruction
        self.conv_trunk = nn.Conv2d(nf, nf, 3, 1, 1)
        
        # Upsampling - només un bloc per scale=2
        self.upconv = self._upconv(nf, nf)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.final = nn.Conv2d(nf, in_channels, kernel_size=1)
        
        self._init_weights()
    
    def _upconv(self, in_ch: int, out_ch: int) -> nn.Module:
        """Upsampling convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        feat = self.lrelu(self.conv_first(x))
        
        # RRDB trunk with residual
        body_feat = self.rrdb_trunk(feat)
        feat = feat + body_feat
        
        # Channel attention
        feat = self.ca(feat)
        
        # Reconstruction
        feat = self.conv_trunk(feat)
        feat = feat + feat  # residual connection
        
        # Upsampling (scale=2)
        feat = self.upconv(feat)
        
        # Output layer
        out = self.final(feat)
        return out


class Discriminator(nn.Module):
    """PatchGAN Discriminator with Spectral Normalization."""
    
    def __init__(self, input_channels: int = 3, nf: int = 64):
        super().__init__()
        
        # Spectral normalization wrapper for stable GAN training
        def conv2d(*args, **kwargs):
            return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))
        
        self.conv = nn.Sequential(
            conv2d(input_channels, nf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            conv2d(nf, nf * 2, 4, 2, 1),
            nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            conv2d(nf * 4, nf * 8, 4, 2, 1),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            conv2d(nf * 8, 1, 4, 1, 0),  # Patch output
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SRGAN(nn.Module):
    """Complete SRGAN model combining Generator and Discriminator."""
    
    def __init__(
        self,
        in_channels: int = 7,
        num_rrdb: int = 16,
        scale: int = 2,
        lr_g: float = 1e-4,
        lr_d: float = 1e-4,
    ):
        super().__init__()
        
        self.generator = Generator(in_channels, num_rrdb, scale=scale)
        self.discriminator = Discriminator(input_channels=3)  # RGB output
        
        self.lr_g = lr_g
        self.lr_d = lr_d
        
        # Loss functions (initialized in training)
        self.criterion_content = nn.L1Loss()
        self.criterion_adversarial = nn.BCEWithLogitsLoss()
    
    def forward_generator(self, lr: torch.Tensor) -> torch.Tensor:
        """Generate super-resolution output."""
        return self.generator(lr)
    
    def forward_discriminator(self, sr: torch.Tensor, hr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Discriminate between real and generated images."""
        fake_out = self.discriminator(sr)
        if hr is not None:
            real_out = self.discriminator(hr)
            return fake_out, real_out
        return fake_out, None
    
    @staticmethod
    def load_from_checkpoint(path: str, device: str = "cpu") -> "SRGAN":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        model = SRGAN(
            in_channels=checkpoint.get("in_channels", 7),
            num_rrdb=checkpoint.get("num_rrdb", 16),
            scale=checkpoint.get("scale", 2),
        )
        model.load_state_dict(checkpoint["generator_state_dict"])
        return model
    
    def save_checkpoint(self, path: str, **kwargs):
        """Save model checkpoint."""
        torch.save({
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "in_channels": 7,
            "num_rrdb": 16,
            "scale": self.generator.scale,
            **kwargs,
        }, path)
