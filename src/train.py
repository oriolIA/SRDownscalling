"""
Training script for SRDownscalling.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import ESRGAN, UNetSR
from .data.wrf_sr_dataset import WRFSuperResDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(output: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute evaluation metrics."""
    mse = nn.MSELoss()(output, target).item()
    mae = nn.L1Loss()(output, target).item()
    
    # PSNR
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    
    return {"mse": mse, "mae": mae, "psnr": psnr.item()}


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    metrics = {"mse": 0, "mae": 0, "psnr": 0}

    for lr, hr in tqdm(loader, desc="Training"):
        lr = lr.to(device)
        hr = hr.to(device)

        optimizer.zero_grad()
        output = model(lr)
        loss = criterion(output, hr)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_metrics = compute_metrics(output.detach(), hr.detach())
        for k, v in batch_metrics.items():
            metrics[k] += v

    n = len(loader)
    return {
        "loss": total_loss / n,
        **{k: v / n for k, v in metrics.items()}
    }


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    metrics = {"mse": 0, "mae": 0, "psnr": 0}

    with torch.no_grad():
        for lr, hr in tqdm(loader, desc="Validating"):
            lr = lr.to(device)
            hr = hr.to(device)

            output = model(lr)
            loss = criterion(output, hr)

            total_loss += loss.item()
            batch_metrics = compute_metrics(output, hr)
            for k, v in batch_metrics.items():
                metrics[k] += v

    n = len(loader)
    return {
        "loss": total_loss / n,
        **{k: v / n for k, v in metrics.items()}
    }


def main():
    parser = argparse.ArgumentParser(description="SRDownscalling Training")
    parser.add_argument("--lr_dir", type=str, default="/home/oriol/data/WRF/1469893/d02")
    parser.add_argument("--hr_dir", type=str, default="/home/oriol/data/WRF/1469893/d05")
    parser.add_argument("--model", type=str, default="unet_sr", choices=["esrgan", "unet_sr"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample_limit", type=int, default=None, help="Limit samples for quick test")
    args = parser.parse_args()

    logger.info(f"Device: {args.device}")

    # Load data
    lr_dir = Path(args.lr_dir)
    hr_dir = Path(args.hr_dir)

    lr_files = sorted(lr_dir.glob("*.nc"))
    hr_files = sorted(hr_dir.glob("*.nc"))

    if args.sample_limit:
        lr_files = lr_files[:args.sample_limit]
        hr_files = hr_files[:args.sample_limit]

    logger.info(f"Found {len(lr_files)} samples")

    # Split
    n_train = int(len(lr_files) * 0.8)
    train_ds = WRFSuperResDataset(lr_files[:n_train], hr_files[:n_train])
    val_ds = WRFSuperResDataset(lr_files[n_train:], hr_files[n_train:])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model
    if args.model == "esrgan":
        model = ESRGAN(in_channels=7, out_channels=2, scale_factor=2)
    else:
        model = UNetSR(in_channels=7, out_channels=2, scale_factor=2)

    model = model.to(args.device)
    logger.info(f"Model: {args.model}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    os.makedirs(args.output_dir, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, args.device)
        val_metrics = validate(model, val_loader, criterion, args.device)

        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, PSNR: {train_metrics['psnr']:.2f}")
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, PSNR: {val_metrics['psnr']:.2f}")

        # Save best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), f"{args.output_dir}/best_model.pth")
            logger.info(f"Saved best model to {args.output_dir}/best_model.pth")

    logger.info(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
