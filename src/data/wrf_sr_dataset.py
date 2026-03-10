"""
WRF Dataset for Super Resolution - Actualitzat per dades reals.
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
    """Dataset for WRF Super Resolution (paired LR-HR).
    
    Dataset: WRF Case 1469893
    - LR (d02): ~3km resolution
    - HR (d05): ~100m resolution
    """

    VARIABLES = ["U", "V", "W", "T", "P", "HGT", "TKE"]
    
    # Normalization statistics (approximate for WRF data)
    NORM_STATS = {
        "U": {"mean": 5.0, "std": 8.0},
        "V": {"mean": 0.0, "std": 6.0},
        "W": {"mean": 0.0, "std": 0.5},
        "T": {"mean": 280.0, "std": 15.0},
        "P": {"mean": 85000.0, "std": 15000.0},
        "HGT": {"mean": 400.0, "std": 350.0},
        "TKE": {"mean": 1.0, "std": 2.0},
    }

    def __init__(
        self,
        lr_files: List[Path],
        hr_files: List[Path],
        input_vars: List[str] = None,
        target_vars: List[str] = ["U", "V"],
        transform = None,
        normalize: bool = True,
        time_slice: Optional[int] = None  # Use single time step
    ):
        """
        Args:
            lr_files: List of low-res NetCDF files (d02)
            hr_files: List of high-res NetCDF files (d05)
            input_vars: Variables for input (LR)
            target_vars: Variables for target (HR)
            transform: Optional transform
            normalize: Whether to normalize data
            time_slice: If set, use only this time index
        """
        self.lr_files = lr_files
        self.hr_files = hr_files
        self.input_vars = input_vars or self.VARIABLES[:2]  # U, V default
        self.target_vars = target_vars
        self.transform = transform
        self.normalize = normalize
        self.time_slice = time_slice
        
        # Get actual dimensions
        self._get_dimensions()
        logger.info(f"Dataset: LR {self.lr_shape}, HR {self.hr_shape}")
        logger.info(f"Files: {len(lr_files)} pairs")

    def _get_dimensions(self):
        """Get spatial dimensions from first file pair."""
        with xr.open_dataset(self.lr_files[0]) as ds:
            self.lr_shape = (ds.dims["lat"], ds.dims["lon"])
            self.n_times = ds.dims["time"]
            self.n_levels = ds.dims["lev"]
        
        with xr.open_dataset(self.hr_files[0]) as ds:
            self.hr_shape = (ds.dims["lat"], ds.dims["lon"])
        
        # Calculate scale factor
        self.scale_factor = (
            self.hr_shape[0] / self.lr_shape[0],
            self.hr_shape[1] / self.lr_shape[1]
        )

    def _load_var(self, file_path: Path, var: str, average_levels: bool = True) -> np.ndarray:
        """Load single variable from NetCDF."""
        with xr.open_dataset(file_path) as ds:
            if var in ds.variables:
                data = ds[var].values
                # Take mean over pressure levels if 4D
                if len(data.shape) == 4:
                    if self.time_slice is not None:
                        data = data[self.time_slice, 0, :, :]  # Single time, surface level
                    else:
                        data = data[:, 0, :, :].mean(axis=0)  # Mean over time and level
                elif len(data.shape) == 3:
                    if self.time_slice is not None:
                        data = data[self.time_slice, :, :]
                    else:
                        data = data.mean(axis=0)  # Mean over time
                
                # Handle NaN values - fill with 0 or interpolate
                if np.isnan(data).any():
                    # Replace NaN with 0 (common approach for WRF land/sea mask)
                    data = np.nan_to_num(data, nan=0.0)
                    
                return data.astype(np.float32)
            return np.zeros(self.lr_shape if "d02" in str(file_path) else self.hr_shape, dtype=np.float32)

    def _normalize(self, data: np.ndarray, var: str) -> np.ndarray:
        """Normalize variable using dataset statistics."""
        stats = self.NORM_STATS.get(var, {"mean": 0, "std": 1})
        data = (data - stats["mean"]) / (stats["std"] + 1e-8)
        return data

    def __len__(self) -> int:
        return len(self.lr_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lr_file = self.lr_files[idx]
        hr_file = self.hr_files[idx]

        # Load LR (d02)
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

        # Convert to tensors
        lr = torch.from_numpy(lr).float()
        hr = torch.from_numpy(hr).float()

        if self.transform:
            lr = self.transform(lr)
            hr = self.transform(hr)

        return lr, hr


def get_file_pairs(lr_dir: Path, hr_dir: Path, limit: int = None) -> Tuple[List[Path], List[Path]]:
    """Get matching LR-HR file pairs by date."""
    lr_files = sorted(lr_dir.glob("*.nc"))
    hr_files = sorted(hr_dir.glob("*.nc"))
    
    # Match by filename pattern (extract date)
    lr_by_date = {f.name.split("_")[-1].replace(".nc", ""): f for f in lr_files}
    hr_by_date = {f.name.split("_")[-1].replace(".nc", ""): f for f in hr_files}
    
    common_dates = sorted(set(lr_by_date.keys()) & set(hr_by_date.keys()))
    
    if limit:
        common_dates = common_dates[:limit]
    
    lr_pairs = [lr_by_date[d] for d in common_dates]
    hr_pairs = [hr_by_date[d] for d in common_dates]
    
    logger.info(f"Matched {len(lr_pairs)} file pairs")
    
    return lr_pairs, hr_pairs


if __name__ == "__main__":
    # Test with actual data
    lr_dir = Path("/home/oriol/data/WRF/1469893/d02")
    hr_dir = Path("/home/oriol/data/WRF/1469893/d05")
    
    lr_files, hr_files = get_file_pairs(lr_dir, hr_dir, limit=5)
    
    print(f"\\nTesting dataset with {len(lr_files)} samples...")
    ds = WRFSuperResDataset(lr_files, hr_files, input_vars=["U", "V"])
    
    lr, hr = ds[0]
    print(f"LR tensor: {lr.shape}")
    print(f"HR tensor: {hr.shape}")
    print(f"Scale factor: {ds.scale_factor}")
    
    # Test single time step
    ds_time = WRFSuperResDataset(lr_files, hr_files, input_vars=["U", "V"], time_slice=0)
    lr_t, hr_t = ds_time[0]
    print(f"\\nWith time_slice=0:")
    print(f"LR: {lr_t.shape}, HR: {hr_t.shape}")
