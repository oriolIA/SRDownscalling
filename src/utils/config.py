"""
Configuration management for SRDownscalling.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
import yaml


@dataclass
class Config:
    """Configuration dataclass for SRDownscalling."""
    
    # Data
    input_dir: str = "data/low_res"
    target_dir: str = "data/high_res"
    scale: int = 2
    batch_size: int = 16
    patch_size: int = 64
    num_workers: int = 4
    train_ratio: float = 0.8
    
    # Model
    in_channels: int = 7
    num_rrdb: int = 16
    nf: int = 64
    gc: int = 32
    
    # Training
    epochs: int = 100
    learning_rate: float = 2e-4
    betas: tuple = (0.9, 0.999)
    
    # Losses
    pixel_weight: float = 1.0
    perceptual_weight: float = 1e-3
    gradient_weight: float = 1e-4
    adversarial_weight: float = 1e-3
    
    # Output
    output_dir: str = "outputs"
    device: str = "cuda"  # or "cpu"
    
    # Experiment name
    experiment_name: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str):
        """Save config to YAML file."""
        data = {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_') and not callable(v)
        }
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    @classmethod
    def default(cls) -> "Config":
        """Create default configuration."""
        return cls()
    
    def save(self, filename: str = "config.yaml"):
        """Save config to file."""
        path = os.path.join(self.output_dir, filename)
        os.makedirs(self.output_dir, exist_ok=True)
        self.to_yaml(path)
        print(f"Config saved to {path}")
