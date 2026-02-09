# SRDownscalling - Working Steps

**Objectiu:** Crear sistema de downscalling amb IA per a dades meteorològiques WRF

## Dataset: WRF Case 1469893

| Domini | Rol | Resolució | Dimensions | Path |
|--------|-----|-----------|------------|------|
| **d02** | INPUT (LR) | ~3km | 48×9×56×57 | `/home/oriol/data/WRF/1469893/d02/` |
| **d05** | OUTPUT (HR) | ~100m | 48×9×125×119 | `/home/oriol/data/WRF/1469893/d05/` |

**Decision:** No tenim d01, així que fem servir **d02 → d05**

**Factors de downscaling:**
- Lat: 56 → 125 (~2.2x)
- Lon: 57 → 119 (~2.1x)
- Temps: 48 hores

**Variables:** TKE, U, V, W, P, T, HGT

---

## Steps

### Step 1: Explorar dades WRF ✅
- [x] Identificar dominis existents
- [x] Mesurar dimensions i resolucions
- [x] Confirmar: **d02 → d05** (decisió: d01 no disponible)

### Step 2: Preparar dades
- [x] Verificar dimensions
- [ ] Crear scripts de preprocessament
- [ ] Generar parells (input, target) d02→d05
- [ ] Normalitzar dades
- [ ] Train/val/test split (70/15/15)

### Step 3: Implementar model
- [x] Implementar ESRGAN (src/models/esrgan.py)
- [x] Implementar UNetSR (src/models/unet_sr.py)
- [ ] Triar arquitectura: **UNetSR** (més estable pel downscaling meteorològic)
- [ ] Entrenar amb dades WRF

### Step 4: Avaluar i optimitzar
- [ ] Mètriques: MSE, SSIM, PSNR
- [ ] Test en dades noves
- [ ] Exportar model

---

## Ús del model

```python
from src.models.unet_sr import UNetSR

model = UNetSR(in_channels=7, out_channels=7)
model.load_checkpoint("checkpoints/sr_model.pth")

# Downscale dades noves
lr_input = load_wrf_domain("d02_path/")
hr_output = model.predict(lr_input)
```

---

## Repo
https://github.com/oriolIA/SRDownscalling
