# SRDownscalling

**Super-Resolution for WRF Wind Fields using Enhanced Deep Learning**

A novel approach combining:
- **ESRGAN-inspired architecture** with residual-in-residual dense blocks
- **Attention mechanisms** (SE + CBAM) for focus on wind features
- **Multi-scale feature fusion** with feature pyramid
- **Perceptual + adversarial + pixel losses** for realistic outputs

## Overview

This project implements a state-of-the-art super-resolution model designed specifically for meteorological downscaling. Unlike traditional UNet approaches, SRDownscalling uses:

1. **RRDB (Residual-in-Residual Dense Block)** - deeper feature extraction
2. **Spectral Normalization GAN** - stable adversarial training
3. **Feature Pyramid Fusion** - multi-scale context
4. **Channel Attention** - adaptive feature recalibration

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/oriolIA/SRDownscalling.git
cd SRDownscalling

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
python -m src.main --mode train \
    --input /path/to/low_res_data/ \
    --target /path/to/high_res_data/ \
    --scale 2 \
    --epochs 100 \
    --batch_size 16
```

### Inference

```bash
python -m src.main --mode predict \
    --model outputs/latest/model.pth \
    --input /path/to/test_low_res.nc \
    --output /path/to/sr_output.nc
```

## Architecture Highlights

```
Input (HxW) → Conv → RRDB×4 → Attention → FPN → Upsample×2 → Output (2Hx2W)
                    ↓
              Discriminator (SNGAN)
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA recommended

See `requirements.txt` for full dependencies.

## License

MIT
