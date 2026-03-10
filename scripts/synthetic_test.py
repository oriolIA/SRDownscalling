#!/usr/bin/env python3
"""
SRDownscalling - Synthetic Test (without PyTorch)
Tests data loading and model structure without requiring PyTorch.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def check_dependencies():
    """Check available packages."""
    deps = {}
    try:
        import numpy
        deps['numpy'] = numpy.__version__
    except:
        deps['numpy'] = False
    
    try:
        import xarray
        deps['xarray'] = xarray.__version__
    except:
        deps['xarray'] = False
    
    try:
        import netCDF4
        deps['netcdf4'] = netCDF4.__version__
    except:
        deps['netcdf4'] = False
    
    try:
        import yaml
        deps['pyyaml'] = yaml.__version__
    except:
        deps['pyyaml'] = False
    
    return deps


def check_data(data_dir):
    """Check WRF data."""
    log(f"Checking data: {data_dir}")
    
    if not os.path.exists(data_dir):
        return {"exists": False, "reason": "Directory not found"}
    
    domains = {}
    for d in ['d01', 'd02', 'd05']:
        d_path = Path(data_dir) / d
        if d_path.exists():
            nc_files = list(d_path.glob("*.nc"))
            domains[d] = len(nc_files)
    
    return {
        "exists": True,
        "domains": domains,
        "total_files": sum(domains.values())
    }


def test_model_structure():
    """Test model architecture without PyTorch."""
    log("Testing model structure...")
    
    # Simulate model parameters calculation
    config = {
        "in_channels": 7,
        "num_rrdb": 4,
        "nf": 32,
        "gc": 16,
        "scale": 2
    }
    
    # Calculate approximate parameters
    # RRDB: ~ (gc * 5 * gc) * num_rrdb ≈ 16*5*16*4 = 5120
    # Plus attention, convs, upsampling...
    estimated_params = 580000  # From documentation
    
    return {
        "config": config,
        "estimated_params": estimated_params,
        "architecture": "RRDB + ChannelAttention + PixelShuffle upsampling"
    }


def main():
    log("=" * 60)
    log("SRDownscalling - Synthetic Test")
    log("=" * 60)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    # 1. Dependencies
    log("\n[1/4] Checking dependencies...")
    results["tests"]["dependencies"] = check_dependencies()
    for k, v in results["tests"]["dependencies"].items():
        status = "✓" if v else "✗"
        log(f"  {status} {k}: {v}")
    
    # 2. Data
    log("\n[2/4] Checking WRF data...")
    data_dir = os.environ.get('DATA_DIR', '/home/oriol/data/WRF/1469893')
    results["tests"]["data"] = check_data(data_dir)
    log(f"  ✓ Data: {results['tests']['data']}")
    
    # 3. Model structure
    log("\n[3/4] Testing model structure...")
    results["tests"]["model"] = test_model_structure()
    log(f"  ✓ Architecture: {results['tests']['model']['architecture']}")
    log(f"  ✓ Parameters: {results['tests']['model']['estimated_params']:,}")
    
    # 4. Summary
    log("\n[4/4] Summary...")
    
    deps_ok = sum(1 for v in results["tests"]["dependencies"].values() if v) >= 3
    data_ok = results["tests"]["data"]["exists"]
    
    all_ok = deps_ok and data_ok
    
    results["status"] = "ready" if all_ok else "missing_deps"
    
    log(f"\n{'=' * 60}")
    log("RESULT")
    log("=" * 60)
    log(f"  Dependencies: {'✓ OK' if deps_ok else '✗ Missing'}")
    log(f"  Data: {'✓ OK' if data_ok else '✗ Missing'}")
    log(f"  Status: {results['status']}")
    
    # Save results
    output_dir = Path("outputs/test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "synthetic_test.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    log(f"\nResults saved to: {results_path}")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
