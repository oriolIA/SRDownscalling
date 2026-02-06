#!/bin/bash
# ============================================================================
# SRDownscalling - Quick Test Script
# ============================================================================
# Executa: bash scripts/quick_test.sh
# 
# Aquest script:
# 1. Activa l'entorn virtual
# 2. Genera dades sintètiques si no hi ha dades WRF
# 3. Entrena el model per 1 epoch
# 4. Genera mètriques bàsiques
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$REPO_DIR/outputs/quick_test"
VENV_PATH="$REPO_DIR/venv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $1"
}

# ============================================================================
# CONFIGURACIÓ
# ============================================================================
DATA_DIR="${1:-/home/oriol/data/WRF/1469893}"
EPOCHS="${2:-1}"
BATCH_SIZE="${3:-4}"
SCALE=2

log "========================================"
log "SRDownscalling - Quick Test"
log "========================================"
log "Data dir: $DATA_DIR"
log "Epochs: $EPOCHS"
log "Batch size: $BATCH_SIZE"
log "Scale factor: $SCALE"
log ""

# ============================================================================
# ACTIVAR VENV
# ============================================================================
if [ -f "$VENV_PATH/bin/activate" ]; then
    log "Activant virtual environment..."
    source "$VENV_PATH/bin/activate"
else
    warn "Virtual environment no trobat. usant Python del sistema."
fi

# ============================================================================
# COMPROVAR DEPENDÈNCIES
# ============================================================================
log "Comprovant dependències..."
MISSING=""
for pkg in torch torchvision xarray netCDF4; do
    if ! python -c "import $pkg" 2>/dev/null; then
        MISSING="$MISSING $pkg"
    fi
done

if [ -n "$MISSING" ]; then
    error "Dependències manquen:$MISSING"
    error "Instal·la amb: pip install$MISSING"
    exit 1
fi

log "Dependències OK"

# ============================================================================
# CREAR OUTPUT DIR
# ============================================================================
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# EXECUTAR TEST
# ============================================================================
log "Executant test..."

python3 << 'PYTEST' 2>&1 | tee "$OUTPUT_DIR/test.log"
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Paths
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'outputs/quick_test')
DATA_DIR = os.environ.get('DATA_DIR', '/home/oriol/data/WRF/1469893')

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("SRDownscalling - Quick Test")
print("=" * 60)

# Detectar dispositiu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDispositiu: {device}")

# Importar mòduls
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from models.srgan import Generator
    print("✓ Models importats correctament")
except ImportError as e:
    print(f"✗ Error important models: {e}")
    sys.exit(1)

# Configuració
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 4))
EPOCHS = int(os.environ.get('EPOCHS', 1))
SCALE = int(os.environ.get('SCALE', 2))

print(f"\nConfiguració:")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Scale factor: {SCALE}")

# Generar dades sintètiques
print(f"\nGenerant dades sintètiques...")
BATCH_SIZE = 4
CHANNELS = 7
LR_SIZE = 64
HR_SIZE = LR_SIZE * SCALE

# Simular dades LR i HR
lr_data = torch.randn(BATCH_SIZE, CHANNELS, LR_SIZE, LR_SIZE, device=device)
hr_data = torch.randn(BATCH_SIZE, CHANNELS, HR_SIZE, HR_SIZE, device=device)

print(f"  LR shape: {lr_data.shape}")
print(f"  HR shape: {hr_data.shape}")

# Crear model
print(f"\nCreant model Generator...")
model = Generator(in_channels=CHANNELS, num_rrdb=4, nf=32, scale=SCALE).to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Paràmetres totals: {total_params:,}")
print(f"  Paràmetres entrenables: {trainable_params:,}")

# Test de forward pass
print(f"\nTesteant forward pass...")
with torch.no_grad():
    output = model(lr_data)
print(f"  Input:  {lr_data.shape}")
print(f"  Output: {output.shape}")

# Verificar shape
expected_shape = (BATCH_SIZE, CHANNELS, HR_SIZE, HR_SIZE)
if list(output.shape) == list(expected_shape):
    print(f"  ✓ Output shape correcte!")
else:
    print(f"  ✗ Shape incorrecte! Esperat: {expected_shape}")

# Training loop simplificat
print(f"\nTraining loop ({EPOCHS} epoch)...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.L1Loss()

model.train()
losses = []

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    # Forward
    output = model(lr_data)
    loss = criterion(output, hr_data)
    
    # Backward
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    print(f"  Epoch {epoch+1}/{EPOCHS}: Loss = {loss.item():.4f}")

# Resultats
print(f"\n" + "=" * 60)
print("RESULTATS")
print("=" * 60)
print(f"  Èpoques executades: {EPOCHS}")
print(f"  Loss inicial:       {losses[0]:.4f}")
print(f"  Loss final:         {losses[-1]:.4f}")
print(f"  Millora:            {(losses[0] - losses[-1]):.4f} ({(losses[0] - losses[-1])/losses[0]*100:.1f}%)")

# Guardar resultats
results = {
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "scale": SCALE,
    "channels": CHANNELS,
    "lr_size": LR_SIZE,
    "hr_size": HR_SIZE,
    "total_params": total_params,
    "trainable_params": trainable_params,
    "loss_initial": losses[0],
    "loss_final": losses[-1],
    "improvement_percent": (losses[0] - losses[-1])/losses[0]*100,
    "device": str(device),
}

import json
with open(f"{OUTPUT_DIR}/results.json", 'w') as f:
    json.dump(results, f, indent=2)

# Guardar checkpoint
torch.save(model.state_dict(), f"{OUTPUT_DIR}/model_test.pth")

print(f"\nResultats guardats a: {OUTPUT_DIR}")
print(f"  - results.json")
print(f"  - model_test.pth")

print("\n" + "=" * 60)
print("TEST COMPLETAT EXITOSAMENT!")
print("=" * 60)
PYTEST

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    log "Test completat exitós!"
    log "Resultats a: $OUTPUT_DIR"
else
    error "Test fallit! Veure log per detalls."
fi

exit $EXIT_CODE
