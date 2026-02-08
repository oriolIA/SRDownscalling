"""
WRF Dataset for Super Resolution.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class WRFSuperResDataset(Dataset):
    """Dataset for WRF Super Resolution (paired LR-HR)."""

    VARIABLES = ["U", "V", "W", "T", "P", "HGT", "TKE"]

    def __init__(
        self,
        lr_files: List[Path],      # Low resolution (d01)
        hr_files: List[Path],       # High resolution (d05)
        input_vars: List[str] = None,
        target_vars: List[str] = ["U", "V"],
        transform = None,
        normalize: bool = True
    ):
        self.lr_files = lr_files
        self.hr_files = hr_files
        self.input_vars = input_vars or self.VARIABLES
        self.target_vars = target_vars
        self.transform = transform
        self.normalize = normalize

        # Load sample to get dimensions
        self._get_dimensions()

    def _get_dimensions(self):
        """Get spatial dimensions from first file."""
        with xr.open_dataset(self.lr_files[0]) as ds:
            self.lr_shape = (
                ds.dims.get("lat", 50),
                ds.dims.get("lon", 51)
            )
        with xr.open_dataset(self.hr_files[0]) as ds:
            self.hr_shape = (
                ds.dims.get("lat", 125),
                ds.dims.get("lon", 119)
            )
        logger.info(f"LR shape: {self.lr_shape}, HR shape: {self.hr_shape}")

    def _load_var(self, file_path: Path, var: str) -> np.ndarray:
        """Load single variable from NetCDF."""
        with xr.open_dataset(file_path) as ds:
            if var in ds.variables:
                data = ds[var].values
                # Average over levels if 4D
                if len(data.shape) == 4:
                    data = data.mean(axis=1)
                return data.astype(np.float32)
            return np.zeros(self.lr_shape, dtype=np.float32)

    def _normalize(self, data: np.ndarray, var: str) -> np.ndarray:
        """Normalize variable."""
        stats = {
            "U": {"mean": 0, "std": 10},
            "V": {"mean": 0, "std": 10},
            "W": {"mean": 0, "std": 5},
            "T": {"mean": 288, "std": 20},
            "P": {"mean": 90000, "std": 10000},
            "HGT": {"mean": 500, "std": 500},
            "TKE": {"mean": 0, "std": 5},
        }.get(var, {"mean": 0, "std": 1})

        data = (data - stats["mean"]) / (stats["std"] + 1e-8)
        return data

    def __len__(self) -> int:
        return len(self.lr_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lr_file = self.lr_files[idx]
        hr_file = self.hr_files[idx]

        # Load LR (d01) - downsample to simulate lower resolution
        lr_data = []
        for var in self.input_vars:
            var_data = self._load_var(lr_file, var)
            if self.normalize:
                var_data = self._normalize(var_data, var)
            lr_data.append(var_data)
        lr = np.stack(lr_data, axis=0)  # (C, H, W)

        # Load HR (d05)
        hr_data = []
        for var in self.target_vars:
            var_data = self._load_var(hr_file, var)
            if self.normalize:
                var_data = self._normalize(var_data, var)
            hr_data.append(var_data)
        hr = np.stack(hr_data, axis=0)  # (C, H, W)

        # Upsample LR to HR size for training
        lr = torch.from_numpy(lr).float()
        hr = torch.from_numpy(hr).float()

        return lr, hr


if __name__ == "__main__":
    # Test
    lr_dir = Path("/home/oriol/data/WRF/1469893/d02")  # Usar d02 com LR
    hr_dir = Path("/home/oriol/data/WRF/1469893/d05")

    lr_files = sorted(lr_dir.glob("*.nc"))[:5]
    hr_files = sorted(hr_dir.glob("*.nc"))[:5]

    ds = WRFSuperResDataset(lr_files, hr_files)
    lr, hr = ds[0]
    print(f"LR: {lr.shape}, HR: {hr.shape}")
