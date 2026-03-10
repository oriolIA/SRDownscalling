"""
SRDownscalling - Super-Resolution for WRF Wind Fields

Main entry point for training and inference.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch

from .data.dataset import WRFDataModule
from .models.srgan import SRGAN
from .training.trainer import SRTrainer
from .utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SRDownscalling: WRF Wind Super-Resolution")
    parser.add_argument("--mode", choices=["train", "predict", "info"], default="info")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--input", type=str, help="Input low-res data directory")
    parser.add_argument("--target", type=str, help="Target high-res data directory")
    parser.add_argument("--model", type=str, help="Path to trained model for prediction")
    parser.add_argument("--output", type=str, help="Output path for predictions")
    parser.add_argument("--scale", type=int, default=2, help="Upscaling factor")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train(args, config: Config):
    """Training pipeline."""
    logger.info(f"Starting training on {args.device}")
    
    # Data module
    data_module = WRFDataModule(
        input_dir=config.input_dir,
        target_dir=config.target_dir,
        scale=config.scale,
        batch_size=config.batch_size,
    )
    data_module.prepare_data()
    data_module.setup()
    
    # Model
    model = SRGAN(
        in_channels=config.in_channels,
        num_rrdb=config.num_rrdb,
        scale=args.scale,
    ).to(args.device)
    
    # Trainer
    trainer = SRTrainer(
        model=model,
        data_module=data_module,
        lr=config.learning_rate,
        device=args.device,
        output_dir=config.output_dir,
    )
    
    trainer.train(epochs=config.epochs)
    
    # Save final model
    trainer.save_checkpoint("srgan_final.pth")
    logger.info(f"Training complete. Model saved to {config.output_dir}")


def predict(args):
    """Inference pipeline."""
    logger.info(f"Running inference with model: {args.model}")
    
    device = torch.device(args.device)
    model = SRGAN.load_from_checkpoint(args.model).to(device)
    model.eval()
    
    # Load data
    data_module = WRFDataModule(input_dir=args.input)
    data_module.prepare_data()
    
    # Generate predictions
    # TODO: implement inference
    logger.info("Inference pipeline ready.")


def main():
    args = parse_args()
    
    if args.mode == "info":
        logger.info("SRDownscalling - WRF Wind Field Super-Resolution")
        logger.info(f"Device: {args.device}")
        logger.info("Use --mode train or --mode predict")
    
    elif args.mode == "train":
        config = Config.load(args.config) if args.config else Config.default()
        train(args, config)
    
    elif args.mode == "predict":
        predict(args)


if __name__ == "__main__":
    main()
