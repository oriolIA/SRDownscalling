# SRDownscalling - Working Steps

**Objectiu:** Crear sistema de downscalling amb IA per a dades meteorològiques WRF

## Dataset: WRF Case 1469893

| Domini | Resolució | Dimensions | Path |
|--------|-----------|------------|------|
| d01 (pare) | ~0.08° (~9km) | 50×51×24×9 | `/home/oriol/data/WRF/1469893/d01/` |
| d05 (fill) | ~0.001° (~100m) | 125×119×48×9 | `/home/oriol/data/WRF/1469893/d05/` |

**Variables:** time, lev (nivells de pressió), lat, lon

**Període:** 2020-01-01 a 2020-12-31 (365 dies)

## Downscalling Target

- **Input:** d01 (~9km)
- **Output:** d05 (~100m)
- **Factor:** ~90× millora espacial

---

## Steps

### Step 1: Explorar dades WRF ✅
- [x] Identificar dominis existents
- [x] Mesurar dimensions i resolucions
- [x] Confirmar: d01 (pare) → d05 (fill)

### Step 2: Preparar dades
- [ ] Crear scripts de preprocessament
- [ ] Generar parells (input, target) d01→d05
- [ ] Normalitzar dades
- [ ] Train/val/test split

### Step 3: Implementar model
- [ ] Implementar arquitectura SR (Super Resolution)
- [ ] Opcions: UNet, ESRGAN, SwinIR, etc.
- [ ] Entrenar amb dades WRF

### Step 4: Avaluar i optimitzar
- [ ] Mètriques: MSE, SSIM, PSNR
- [ ] Test en dades noves

---

## Repo
https://github.com/oriol/SRDownscalling
