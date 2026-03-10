"""
WRF Data Module - Load and preprocess WRF NetCDF data for super-resolution.

Handles:
- Loading low-res and high-res NetCDF files
- Normalization of meteorological variables
- Creating paired samples for training
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import xarray as xr


class WRFNormalizer:
    """Normalize WRF variables for neural network input."""
    
    # Variable-specific normalization statistics
    STATS = {
        "U": {"mean": 0.0, "std": 10.0, "clip": (-30, 30)},
        "V": {"mean": 0.0, "std": 10.0, "clip": (-30, 30)},
        "W": {"mean": 0.0, "std": 5.0, "clip": (-15, 15)},
        "T": {"mean": 288.0, "std": 20.0, "clip": (250, 330)},
        "P": {"mean": 90000.0, "std": 10000.0, "clip": (50000, 110000)},
        "HGT": {"mean": 500.0, "std": 500.0, "clip": (0, 4000)},
        "TKE": {"mean": 0.0, "std": 5.0, "clip": (0, 25)},
    }
    
    # Default for unknown variables
    DEFAULT = {"mean": 0.0, "std": 1.0, "clip": None}
    
    @classmethod
    def normalize(cls, var_name: str, data: np.ndarray) -> np.ndarray:
        """Normalize a single variable."""
        stats = cls.STATS.get(var_name, cls.DEFAULT)
        
        # Clip if specified
        if stats["clip"] is not None:
            data = np.clip(data, stats["clip"][0], stats["clip"][1])
        
        # Z-score normalization
        data = (data - stats["mean"]) / (stats["std"] + 1e-8)
        return data
    
    @classmethod
    def denormalize(cls, var_name: str, data: np.ndarray) -> np.ndarray:
        """Denormalize a single variable."""
        stats = cls.STATS.get(var_name, cls.DEFAULT)
        return data * stats["std"] + stats["mean"]
    
    @classmethod
    def get_default_scale(cls) -> float:
        """Get default scale factor for upsampling."""
        return 2.0


class WRFDataset(Dataset):
    """Dataset for WRF super-resolution training."""
    
    VARIABLES = ["U", "V", "W", "T", "P", "HGT", "TKE"]
    
    def __init__(
        self,
        input_files: List[str],
        target_files: Optional[List[str]] = None,
        scale: int = 2,
        patch_size: int = 64,
        transform: Optional[callable] = None,
        is_train: bool = True,
    ):
        """
        Args:
            input_files: List of low-res NetCDF file paths
            target_files: List of high-res NetCDF file paths (optional, for inference)
            scale: Upscaling factor
            patch_size: Random crop size for training
            transform: Optional data augmentation
            is_train: Training or inference mode
        """
        self.input_files = input_files
        self.target_files = target_files or input_files
        self.scale = scale
        self.patch_size = patch_size
        self.transform = transform
        self.is_train = is_train
        
        # Load metadata from first file
        self._load_metadata()
    
    def _load_metadata(self):
        """Load spatial dimensions from first file."""
        with xr.open_dataset(self.input_files[0]) as ds:
            self.lat_size = ds.dims["lat"]
            self.lon_size = ds.dims["lon"]
            self.time_size = ds.dims["time"]
            self.levels = ds.dims.get("lev", 1)
    
    def _load_sample(self, file_path: str) -> np.ndarray:
        """Load all variables from a NetCDF file."""
        with xr.open_dataset(file_path) as ds:
            # Stack all variables into channels
            variables = []
            for var in self.VARIABLES:
                if var in ds.variables:
                    var_data = ds[var].values
                    # Average over levels if multi-level
                    if len(var_data.shape) == 4:  # (time, lev, lat, lon)
                        var_data = var_data.mean(axis=1)  # (time, lat, lon)
                    variables.append(var_data)
                else:
                    # Fill with zeros if variable missing
                    shape = (self.time_size, self.lat_size, self.lon_size)
                    variables.append(np.zeros(shape, dtype=np.float32))
            
            return np.stack(variables, axis=-1)  # (time, lat, lon, ch)
    
    def _downsample(self, data: np.ndarray, scale: int) -> np.ndarray:
        """Downsample using bicubic interpolation."""
        # Data: (time, lat, lon, ch) -> (time, ch, lat, lon) for torch
        data = data.transpose(0, 3, 1, 2)
        
        # Downsample
        scaled_size = (data.shape[2] // scale, data.shape[3] // scale)
        low_res = F.interpolate(
            torch.from_numpy(data).float(),
            size=scaled_size,
            mode="bicubic",
            align_corners=True,
        ).numpy()
        
        # Back to (time, lat, lon, ch)
        return low_res.transpose(0, 2, 3, 1)
    
    def __len__(self) -> int:
        return len(self.input_files) * self.time_size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        file_idx = idx // self.time_size
        time_idx = idx % self.time_size
        
        # Load input (low-res) and target (high-res)
        input_data = self._load_sample(self.input_files[file_idx])
        target_data = self._load_sample(self.target_files[file_idx])
        
        # Extract single time step
        lr = input_data[time_idx]  # (lat, lon, ch)
        hr = target_data[time_idx]
        
        # Normalize
        for i, var in enumerate(self.VARIABLES):
            lr[..., i] = WRFNormalizer.normalize(var, lr[..., i])
            hr[..., i] = WRFNormalizer.normalize(var, hr[..., i])
        
        # Convert to tensors (ch, lat, lon)
        lr = torch.from_numpy(lr.transpose(2, 0, 1)).float()
        hr = torch.from_numpy(hr.transpose(2, 0, 1)).float()
        
        # Training: random crop
        if self.is_train and self.patch_size > 0:
            _, lr_h, lr_w = lr.shape
            
            # Random crop coordinates
            lr_crop_h = random.randint(0, lr_h - self.patch_size)
            lr_crop_w = random.randint(0, lr_w - self.patch_size)
            
            hr_crop_h = lr_crop_h * self.scale
            hr_crop_w = lr_crop_w * self.scale
            
            lr = lr[:, lr_crop_h:lr_crop_h + self.patch_size, lr_crop_w:lr_crop_w + self.patch_size]
            hr = hr[:, hr_crop_h:hr_crop_h + self.patch_size * self.scale, 
                     hr_crop_w:hr_crop_w + self.patch_size * self.scale]
        
        return {
            "lr": lr,  # (ch, H, W)
            "hr": hr,  # (ch, scale*H, scale*W)
            "file": os.path.basename(self.input_files[file_idx]),
            "time_idx": time_idx,
        }


class WRFDataModule:
    """Data module for WRF super-resolution."""
    
    def __init__(
        self,
        input_dir: str,
        target_dir: Optional[str] = None,
        scale: int = 2,
        batch_size: int = 16,
        patch_size: int = 64,
        num_workers: int = 4,
        train_ratio: float = 0.8,
    ):
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir) if target_dir else input_dir
        self.scale = scale
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        
        self.input_files: List[str] = []
        self.train_dataset: Optional[WRFDataset] = None
        self.val_dataset: Optional[WRFDataset] = None
        self.test_dataset: Optional[WRFDataset] = None
    
    def prepare_data(self):
        """Discover and organize data files."""
        # Find all NetCDF files
        self.input_files = sorted(self.input_dir.glob("*.nc"))
        print(f"Found {len(self.input_files)} NetCDF files in {self.input_dir}")
        
        if len(self.input_files) == 0:
            raise ValueError(f"No NetCDF files found in {self.input_dir}")
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/validation/testing."""
        if len(self.input_files) == 0:
            self.prepare_data()
        
        # Split files
        n_files = len(self.input_files)
        n_train = int(n_files * self.train_ratio)
        n_val = int(n_files * (1 - self.train_ratio) / 2)
        
        train_files = self.input_files[:n_train]
        val_files = self.input_files[n_train:n_train + n_val]
        test_files = self.input_files[n_train + n_val:]
        
        # Create datasets
        self.train_dataset = WRFDataset(
            train_files, self.target_dir, self.scale,
            self.patch_size, is_train=True,
        )
        self.val_dataset = WRFDataset(
            val_files, self.target_dir, self.scale,
            patch_size=0, is_train=False,
        )
        self.test_dataset = WRFDataset(
            test_files, self.target_dir, self.scale,
            patch_size=0, is_train=False,
        )
        
        print(f"Train: {len(self.train_dataset)} samples")
        print(f"Val: {len(self.val_dataset)} samples")
        print(f"Test: {len(self.test_dataset)} samples")
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
