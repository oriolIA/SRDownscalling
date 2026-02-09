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

## Autor

Oriol
