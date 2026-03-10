# SRDownscalling - WRF Wind Field Super-Resolution

Super-Resolution models per a dades meteorològiques WRF (d02 → d05).

## Objectiu

Aplicar tècniques de Deep Learning (ESRGAN + Attention) per fer downscaling de camps de vent des de ~3km (d02) fins a ~100m (d05).

## Dataset

**WRF Case:** 1469893

| Domini | Resolució | Dimensions | Variables |
|--------|-----------|------------|-----------|
| d02 (LR) | ~3km | 48×9×56×57 | TKE, U, V, W, P, T, HGT |
| d05 (HR) | ~100m | 48×9×125×119 | TKE, U, V, W, P, T, HGT |

## Models

### UNetSR (Principal)
- Basat en U-Net amb residual connections
- Attention gates per capturar patrons de vent
- ~2.5M paràmetres

### ESRGAN (Experimental)
- Super-Resolution amb GAN
- Residual-in-Residual Dense Blocks
- ~680K paràmetres

---

## 🆕 Noves Recerques

### Recerca de Models (vegeu `RESEARCH_MODELS.md`)

| Model | Tipus | Parameters | 3D Suport | Recomanat |
|-------|-------|------------|-----------|-----------|
| **3D UNet + CBAM** | Híbrid CNN-Transformer | 8-12M | ✅ | ⭐ **Millor opció** |
| VNet3D | Volumètric | 15-20M | ✅ | Per a dades 3D |
| SwinUNetR3D | Transformer | 20M+ | ✅ | Per a llarg abast |
| Diffusion Models | Generatiu | Variable | ❌ | Alta qualitat, lent |

**Recomanació principal:** 3D UNet amb ResNet encoder + CBAM attention per balancejar rendiment i eficiència.

### Recerca de Loss Functions (vegeu `RESEARCH_LOSS_FUNCTIONS.md`)

| Loss | Ús | Impacte |
|------|-----|---------|
| L1 + SSIM | Baseline | +5% SSIM vs L1 només |
| Combined (L1+SSIM+Gradient+Frequency) | Recomanat | +10% SSIM |
| Physics-informed | Preservar física | +5% precisió física |

**Recomanació:** CombinedLoss amb:
- L1: 1.0 (reconstrucció principal)
- SSIM: 0.5 (similaritat estructural)
- Gradient: 0.1 (preservar edges)
- Frequency: 0.1 (consistència espectral)

### Recerca de Feature Engineering (vegeu `RESEARCH_FEATURE_ENGINEERING.md`)

**Features recomanades per a camps de vent:**

1. **Components de vent:** U, V (existents)
2. **Velocitat del vent:** √(U² + V²) → +5-10% SSIM
3. **Direcció:** sin/cos(atan2(U,V)) → +3-5%
4. **Curl/Vorticitat:** ∂v/∂x - ∂u/∂y → +3-5%
5. **Divergència:** ∂u/∂x + ∂v/∂y → +3-5%
6. **TKE:** Turbulent Kinetic Energy → +5-8%
7. **Features temporals (4D):** Per a seqüències → +10-15%

### Experiments Proposats (vegeu `EXPERIMENTS.md`)

| Experiment | Durada | Objectiu |
|------------|--------|----------|
| 1. Baseline | 2h | Metrics de referència |
| 2. 3D Models | 8h | Avantatge volumètric |
| 3. Loss Functions | 6h | Configuració òptima |
| 4. Features | 4h | Impacte de features |
| 5. Generalization | 2h | Robustesa domini |
| 6. Hyperparameter | 24h | Optimització |
| 7. Ensemble | 1h | Millora final |

---

## Estructura del Projecte

```
SRDownscalling/
├── src/
│   ├── models/          # Models (UNetSR, ESRGAN, VNet3D, etc.)
│   ├── data/            # Dataset loaders
│   ├── training/        # Training utilities
│   ├── features/        # Feature engineering
│   └── utils/           # Utilities
├── data/                # Dades WRF
├── outputs/             # Resultats
├── RESEARCH_MODELS.md   # Recerca de models
├── RESEARCH_LOSS_FUNCTIONS.md  # Recerca de pèrdues
├── RESEARCH_FEATURE_ENGINEERING.md  # Recerca de features
├── EXPERIMENTS.md       # Experiments proposats
└── README.md
```

## Ús

```bash
# Entrenar
python src/train.py --model unetsr --epochs 100 --batch 4

# Inferència
python src/inference.py --model checkpoints/unetsr_final.pth --input d02_sample.nc
```

## Requisits

- Python 3.11+
- PyTorch 2.0+
- xarray, netCDF4, numpy, scipy

## Mètriques Esperades

| Mètrica | Valor Objectiu |
|---------|----------------|
| MAE | < 0.5 m/s |
| RMSE | < 1.0 m/s |
| SSIM | > 0.85 |
| Correlation | > 0.90 |

## Referències

1. Milletari et al. (2016). "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation."
2. Liu et al. (2021). "SwinIR: Image Restoration Using Swin Transformer."
3. Wang et al. (2004). "Image Quality Assessment: From Error Visibility to Structural Similarity."
4. Ledig et al. (2017). "Photo-Realistic Single Image Super-Resolution Using a GAN."
5. Hatamizadeh et al. (2022). "SwinUNETR: Swin Transformer for Volumetric Medical Image Segmentation."

## Autor

Oriol
