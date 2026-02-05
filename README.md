# SRDownscalling - Super Resolution per a Dades WRF

**Repo:** https://github.com/oriol/SRDownscalling

## Objectiu

Sistema de downscalling amb IA per a dades meteorològiques WRF (Weather Research and Forecasting).

## Dataset

**Font:** `/home/oriol/data/WRF/1469893`

| Domini | Resolució | Dimensions | Temporal |
|--------|-----------|------------|----------|
| d01 (pare) | ~9km (0.08°) | 50×51 | 24h |
| d05 (fill) | ~100m (0.001°) | 125×119 | 24h |

**Factor de millora:** ~90× espacial

## Instal·lació

```bash
git clone https://github.com/oriol/SRDownscalling.git
cd SRDownscalling
pip install -r requirements.txt
```

## Ús

```bash
# Prova ràpida amb dades reduïdes
bash scripts/quick_test.sh

# Entrenament complet
python src/train.py --config configs/sr_resunet.yaml

# Inferència
python src/predict.py --model outputs/best_model.pth --input data/test/
```

## Estructura

```
SRDownscalling/
├── src/
│   ├── models/       # Arquitectes SR (ESRGAN, SwinIR, UNet)
│   ├── data/         # Dataset WRF
│   ├── training/      # Pipeline d'entrenament
│   └── utils/        # Utilitats
├── configs/          # Configuracions YAML
├── scripts/          # Scripts de prova
└── data/             # Dades (enllaç simbòlic)
```

## Model

Arquitectura inicial: **ESRGAN** (Enhanced Super-Resolution GAN) adaptada per a dades meteorològiques.

## Resultats Esperats

| Mètrica | Valor estimat (inicial) | Objectiu |
|---------|------------------------|----------|
| PSNR | 25-28 dB | >32 dB |
| SSIM | 0.70-0.75 | >0.85 |
| MAE | 1.5-2.0 m/s | <1.0 m/s |

## Autor

Oriol
