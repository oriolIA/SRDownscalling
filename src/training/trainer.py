"""
Training utilities for SRGAN.

Includes:
- Perceptual loss (VGG-based)
- Combined loss functions
- Training loop
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
from tqdm import tqdm
import numpy as np


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features."""
    
    def __init__(self, device: str = "cuda", weight: float = 1e-3):
        super().__init__()
        self.weight = weight
        self.vgg = self._build_vgg().to(device)
        self.vgg.eval()
        
        # Freeze VGG
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def _build_vgg(self) -> nn.Module:
        """Build VGG19 for perceptual loss."""
        import torchvision.models as models
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # Use features up to conv4_4 for perceptual loss
        return nn.Sequential(*list(vgg.features)[:20])
    
    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss between SR and HR images."""
        # Convert to 3-channel for VGG (use first 3 channels or replicate)
        if sr.shape[1] > 3:
            sr_vgg = sr[:, :3, :, :]
            hr_vgg = hr[:, :3, :, :]
        else:
            sr_vgg = sr
            hr_vgg = hr
        
        # Normalize for VGG
        sr_vgg = (sr_vgg + 1) / 2  # [-1, 1] -> [0, 1]
        hr_vgg = (hr_vgg + 1) / 2
        
        sr_features = self.vgg(sr_vgg)
        hr_features = self.vgg(hr_vgg)
        
        return F.mse_loss(sr_features, hr_features) * self.weight


class GradientLoss(nn.Module):
    """Gradient-aware loss for sharp edges."""
    
    def __init__(self, weight: float = 1e-4):
        super().__init__()
        self.weight = weight
    
    def _gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute image gradients using Sobel operator."""
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                dtype=x.dtype, device=x.device).float()
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                dtype=x.dtype, device=x.device).float()
        
        kernel_x = kernel_x.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
        kernel_y = kernel_y.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
        
        grad_x = F.conv2d(x, kernel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, kernel_y, padding=1, groups=x.shape[1])
        
        return grad_x, grad_y
    
    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_x, sr_y = self._gradient(sr)
        hr_x, hr_y = self._gradient(hr)
        
        loss_x = F.mse_loss(sr_x, hr_x)
        loss_y = F.mse_loss(sr_y, hr_y)
        
        return (loss_x + loss_y) * self.weight


class SRLoss(nn.Module):
    """Combined loss for super-resolution training."""
    
    def __init__(
        self,
        pixel_weight: float = 1.0,
        perceptual_weight: float = 1e-3,
        gradient_weight: float = 1e-4,
        adversarial_weight: float = 1e-3,
        device: str = "cuda",
    ):
        super().__init__()
        self.pixel_weight = pixel_weight
        self.adversarial_weight = adversarial_weight
        
        self.pixel_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss(weight=perceptual_weight, device=device)
        self.gradient_loss = GradientLoss(weight=gradient_weight)
    
    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        discriminator_out: Optional[torch.Tensor] = None,
        real_out: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss."""
        losses = {}
        
        # Pixel-wise loss (L1)
        losses["pixel"] = self.pixel_loss(sr, hr) * self.pixel_weight
        
        # Perceptual loss
        losses["perceptual"] = self.perceptual_loss(sr, hr)
        
        # Gradient loss
        losses["gradient"] = self.gradient_loss(sr, hr)
        
        # Adversarial loss (for generator)
        if discriminator_out is not None:
            losses["adversarial"] = F.binary_cross_entropy_with_logits(
                discriminator_out, 
                torch.ones_like(discriminator_out)
            ) * self.adversarial_weight
        
        # Total loss
        losses["total"] = sum(losses.values())
        
        return losses


class SRTrainer:
    """Training manager for SRGAN."""
    
    def __init__(
        self,
        model: nn.Module,
        data_module,
        lr_g: float = 2e-4,
        lr_d: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        device: str = "cuda",
        output_dir: str = "outputs",
    ):
        self.model = model
        self.data_module = data_module
        self.device = torch.device(device)
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Optimizers
        self.optimizer_g = torch.optim.Adam(
            self.model.generator.parameters(), lr=lr_g, betas=betas
        )
        self.optimizer_d = torch.optim.Adam(
            self.model.discriminator.parameters(), lr=lr_d, betas=betas
        )
        
        # Learning rate schedulers
        self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g, T_max=100, eta_min=1e-7
        )
        self.scheduler_d = torch.optim.lr_scheduler.CosineAnneilingLR(
            self.optimizer_d, T_max=100, eta_min=1e-7
        )
        
        # Loss function
        self.criterion = SRLoss(device=device)
        
        # Metrics
        self.best_psnr = 0
        self.history = {"train": [], "val": []}
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {"total": 0, "pixel": 0, "perceptual": 0, "gradient": 0, "adversarial": 0, "d_loss": 0}
        
        train_loader = self.data_module.train_dataloader()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            lr = batch["lr"].to(self.device)
            hr = batch["hr"].to(self.device)
            
            # ============ Discriminator ============
            self.optimizer_d.zero_grad()
            
            # Real images
            real_out = self.model.forward_discriminator(hr)
            # Fake images
            with torch.no_grad():
                sr_fake = self.model.forward_generator(lr)
            fake_out = self.model.forward_discriminator(sr_fake.detach())
            
            # D loss
            d_loss = F.binary_cross_entropy_with_logits(
                real_out, torch.ones_like(real_out)
            ) + F.binary_cross_entropy_with_logits(
                fake_out, torch.zeros_like(fake_out)
            )
            d_loss.backward()
            self.optimizer_d.step()
            
            # ============ Generator ============
            self.optimizer_g.zero_grad()
            
            sr = self.model.forward_generator(lr)
            fake_out = self.model.forward_discriminator(sr)
            
            # G losses
            losses = self.criterion(sr, hr, fake_out)
            losses["total"].backward()
            self.optimizer_g.step()
            
            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
                elif key == "d_loss":
                    epoch_losses[key] += d_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                "G": f"{losses['total'].item():.4f}",
                "D": f"{d_loss.item():.4f}",
            })
        
        # Average losses
        n_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validation step."""
        self.model.eval()
        val_losses = {"psnr": 0, "ssim": 0, "pixel": 0}
        
        val_loader = self.data_module.val_dataloader()
        
        for batch in val_loader:
            lr = batch["lr"].to(self.device)
            hr = batch["hr"].to(self.device)
            
            sr = self.model.forward_generator(lr)
            
            # PSNR
            mse = F.mse_loss(sr, hr).item()
            psnr = 10 * np.log10(1.0 / (mse + 1e-8))
            val_losses["psnr"] += psnr
            
            # SSIM (simplified)
            val_losses["ssim"] += self._ssim(sr, hr)
            val_losses["pixel"] += F.l1_loss(sr, hr).item()
        
        n_batches = len(val_loader)
        val_losses["psnr"] /= n_batches
        val_losses["ssim"] /= n_batches
        val_losses["pixel"] /= n_batches
        
        if val_losses["psnr"] > self.best_psnr:
            self.best_psnr = val_losses["psnr"]
            self.save_checkpoint("srgan_best.pth")
        
        return val_losses
    
    def _ssim(self, sr: torch.Tensor, hr: torch.Tensor) -> float:
        """Compute SSIM (simplified)."""
        # Use first channel for SSIM calculation
        sr_ch = sr[:, :1, :, :]
        hr_ch = hr[:, :1, :, :]
        
        mu_x = sr_ch.mean()
        mu_y = hr_ch.mean()
        
        sigma_x = sr_ch.var()
        sigma_y = hr_ch.var()
        sigma_xy = ((sr_ch - mu_x) * (hr_ch - mu_y)).mean()
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        )
    
    def train(self, epochs: int = 100):
        """Full training loop."""
        for epoch in range(epochs):
            train_losses = self.train_epoch(epoch)
            val_losses = self.validate(epoch)
            
            # Log
            print(f"\nEpoch {epoch}:")
            print(f"  Train - G: {train_losses['total']:.4f}, D: {train_losses['d_loss']:.4f}")
            print(f"  Val   - PSNR: {val_losses['psnr']:.2f} dB, SSIM: {val_losses['ssim']:.4f}")
            
            # Learning rate update
            self.scheduler_g.step()
            self.scheduler_d.step()
            
            # Save checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f"srgan_epoch_{epoch}.pth")
            
            # History
            self.history["train"].append(train_losses)
            self.history["val"].append(val_losses)
        
        # Save final
        self.save_checkpoint("srgan_final.pth")
        print(f"\nTraining complete. Best PSNR: {self.best_psnr:.2f} dB")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.output_dir, filename)
        self.model.save_checkpoint(path)
        print(f"Saved checkpoint: {path}")
