"""
RRDB (Residual-in-Residual Dense Block) - Core building block for SRGAN

Based on ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block with 5 convolutional layers."""
    
    def __init__(self, nf: int = 64, gc: int = 32, bias: bool = True):
        super().__init__()
        # gc: growth channel
        self.conv1 = nn.Conv2d(nf + 0 * gc, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + 1 * gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block."""
    
    def __init__(self, nf: int = 64, gc: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)
    
    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention."""
    
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """Channel Attention Module (simplified CBAM)."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.se = SEBlock(channels, reduction)
    
    def forward(self, x):
        return self.se(x)


class ResidualRRDB(nn.Module):
    """RRDB with residual connection and attention."""
    
    def __init__(self, nf: int = 64, gc: int = 32):
        super().__init__()
        self.rrdb = RRDB(nf, gc)
        self.ca = ChannelAttention(nf)
    
    def forward(self, x):
        out = self.rrdb(x)
        out = self.ca(out)
        return out * 0.2 + x
