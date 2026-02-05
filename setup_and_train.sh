#!/bin/bash
# ============================================================================
# SRDownscalling - Complete Setup, Training & Deployment Script
# ============================================================================
# Executa: bash setup_and_train_srdownscalling.sh
# ============================================================================

set -e
set -o pipefail

LOG_FILE="/home/oriol/.openclaw/workspace/git/SRDownscalling/outputs/setup.log"
mkdir -p /home/oriol/.openclaw/workspace/git/SRDownscalling/outputs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

error() {
    echo "[ERROR] $*" | tee -a "$LOG_FILE"
    exit 1
}

# ============================================================================
# CONFIGURACIÓ
# ============================================================================
REPO_DIR="/home/oriol/.openclaw/workspace/git/SRDownscalling"
LR_DIR="/home/oriol/data/WRF/1469893/d01"
HR_DIR="/home/oriol/data/WRF/1469893/d05"
GPU_SERVER="oriol@192.168.1.100"  # Canviar per la teva IP
GPU_DIR="/home/oriol/gpu_projects/SRDownscalling"
VENV_PATH="/home/oriol/.openclaw/workspace/venv_sr_test"

log "========================================"
log "SRDownscalling - Setup & Training"
log "========================================"

# ============================================================================
# 1. PREPARAR REPO I DEPENDÈNCIES
# ============================================================================
log "[1/6] Preparant entorn..."

cd "$REPO_DIR"

# Instal·lar dependències
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    pip install --quiet torch torchvision xarray netCDF4 tqdm pyyaml scikit-image
else
    error "Virtual environment no trobat: $VENV_PATH"
fi

# ============================================================================
# 2. CREAR ESTRUCTURA COMPLETA DEL REPO
# ============================================================================
log "[2/6] Creant estructura del repo..."

mkdir -p src/{models,data,training,utils} configs scripts data outputs

# Models
cat > src/models/__init__.py << 'EOF'
from .esrgan import ESRGAN
from .unet_sr import UNetSR

__all__ = ['ESRGAN', 'UNetSR']
EOF

cat > src/models/esrgan.py << 'PYEOF'
"""ESRGAN Super Resolution Model."""
import torch
import torch.nn as nn

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_ch=64, growth_ch=32, num_layers=5):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_ch + i * growth_ch, growth_ch, 3, padding=1) 
            for i in range(num_layers)
        ])
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv_out = nn.Conv2d(in_ch + num_layers * growth_ch, 64, 3, padding=1)

    def forward(self, x):
        inputs = [x]
        for layer in self.layers:
            out = layer(torch.cat(inputs, dim=1))
            inputs.append(self.lrelu(out))
        return self.conv_out(torch.cat(inputs, dim=1)) + x

class RRDB(nn.Module):
    def __init__(self, in_ch=64):
        super().__init__()
        self.rdbs = nn.ModuleList([ResidualDenseBlock() for _ in range(3)])

    def forward(self, x):
        out = x
        for rdb in self.rdbs:
            out = rdb(out)
        return out * 0.2 + x

class ESRGAN(nn.Module):
    def __init__(self, in_ch=7, out_ch=2, num_rrdb=12, scale=2):
        super().__init__()
        self.conv_first = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.rrdb_trunk = nn.Sequential(*[RRDB() for _ in range(num_rrdb)])
        
        self.upscale = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.PixelShuffle(2),
        )
        self.conv_last = nn.Conv2d(64, out_ch, 3, padding=1)

    def forward(self, x):
        x = self.lrelu(self.conv_first(x))
        x = self.rrdb_trunk(x)
        x = self.upscale(x)
        return self.conv_last(x)
PYEOF

cat > src/models/unet_sr.py << 'PYEOF'
"""UNet-based Super Resolution."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UNetSR(nn.Module):
    def __init__(self, in_ch=7, out_ch=2, n_filters=64, scale=2):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_ch, n_filters, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc1 = DoubleConv(n_filters, n_filters)
        self.enc2 = DoubleConv(n_filters, n_filters*2)
        self.enc3 = DoubleConv(n_filters*2, n_filters*4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(n_filters*4, n_filters*8)
        self.up3 = nn.ConvTranspose2d(n_filters*8, n_filters*4, 2, stride=2)
        self.dec3 = DoubleConv(n_filters*8, n_filters*4)
        self.up2 = nn.ConvTranspose2d(n_filters*4, n_filters*2, 2, stride=2)
        self.dec2 = DoubleConv(n_filters*4, n_filters*2)
        self.up1 = nn.ConvTranspose2d(n_filters*2, n_filters, 2, stride=2)
        self.dec1 = DoubleConv(n_filters*2, n_filters)
        self.final = nn.Conv2d(n_filters, out_ch, 1)

    def forward(self, x):
        x = self.init_conv(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([F.interpolate(d3, e3.shape[2:], mode='bilinear'), e3], 1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([F.interpolate(d2, e2.shape[2:], mode='bilinear'), e2], 1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([F.interpolate(d1, e1.shape[2:], mode='bilinear'), e1], 1))
        return self.final(d1)
PYEOF

# Data
cat > src/data/wrf_sr_dataset.py << 'PYEOF'
"""WRF SR Dataset."""
import logging
from pathlib import Path
import numpy as np
import torch
import xarray as xr

logger = logging.getLogger(__name__)

class WRFSuperResDataset(torch.utils.data.Dataset):
    VARIABLES = ["U", "V", "W", "T", "P", "HGT", "TKE"]
    
    def __init__(self, lr_files, hr_files, input_vars=None, target_vars=["U", "V"]):
        self.lr_files = lr_files
        self.hr_files = hr_files
        self.input_vars = input_vars or self.VARIABLES
        self.target_vars = target_vars
        
        # Get dimensions
        with xr.open_dataset(self.lr_files[0]) as ds:
            self.lr_shape = (ds.dims.get("lat", 50), ds.dims.get("lon", 51))
        with xr.open_dataset(self.hr_files[0]) as ds:
            self.hr_shape = (ds.dims.get("lat", 125), ds.dims.get("lon", 119))
        logger.info(f"LR: {self.lr_shape}, HR: {self.hr_shape}")

    def _norm(self, data, var):
        stats = {"U":(0,10), "V":(0,10), "W":(0,5), "T":(288,20), "P":(90000,10000), "HGT":(500,500), "TKE":(0,5)}.get(var, (0,1))
        return (data - stats[0]) / stats[1]

    def __len__(self): return len(self.lr_files)

    def __getitem__(self, idx):
        lr, hr = [], []
        for var in self.input_vars:
            with xr.open_dataset(self.lr_files[idx]) as ds:
                d = ds[var].values.mean(axis=1) if len(ds[var].shape)==4 else ds[var].values
                lr.append(self._norm(d.astype(np.float32), var))
        for var in self.target_vars:
            with xr.open_dataset(self.hr_files[idx]) as ds:
                d = ds[var].values.mean(axis=1) if len(ds[var].shape)==4 else ds[var].values
                hr.append(self._norm(d.astype(np.float32), var))
        return torch.from_numpy(np.stack(lr)), torch.from_numpy(np.stack(hr))
PYEOF

# Training
cat > src/train.py << 'PYEOF'
"""Training script."""
import argparse, logging, os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def metrics(pred, target):
    mse = nn.MSELoss()(pred, target).item()
    mae = nn.L1Loss()(pred, target).item()
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    return {"mse": mse, "mae": mae, "psnr": psnr}

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_dir', default='/home/oriol/data/WRF/1469893/d01')
    parser.add_argument('--hr_dir', default='/home/oriol/data/WRF/1469893/d05')
    parser.add_argument('--model', default='unet_sr')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--samples', type=int, default=None)
    args = parser.parse_args()

    from models import UNetSR
    from data.wrf_sr_dataset import WRFSuperResDataset

    lr_files = sorted(Path(args.lr_dir).glob('*.nc'))[:args.samples]
    hr_files = sorted(Path(args.hr_dir).glob('*.nc'))[:args.samples]
    n_train = int(len(lr_files) * 0.8)

    train_ds = WRFSuperResDataset(lr_files[:n_train], hr_files[:n_train])
    val_ds = WRFSuperResDataset(lr_files[n_train:], hr_files[n_train:])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = UNetSR(in_channels=7, out_channels=2, n_filters=64)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Model: {args.model}, params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for lr, hr in tqdm(train_loader, desc=f"Epoch {epoch}"):
            lr, hr = lr.to(model.device), hr.to(model.device)
            optimizer.zero_grad()
            out = model(lr)
            loss = criterion(out, hr)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for lr, hr in val_loader:
                lr, hr = lr.to(model.device), hr.to(model.device)
                out = model(lr)
                val_loss += criterion(out, hr).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        m = metrics(out, hr)

        logger.info(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, PSNR={m['psnr']:.2f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"{args.output_dir}/best_model.pth")

    logger.info(f"Training complete. Best val loss: {best_loss:.4f}")

if __name__ == '__main__':
    train()
PYEOF

log "Estructura creada"

# ============================================================================
# 3. CONFIGURACIONS I SCRIPTS
# ============================================================================
log "[3/6] Creant configuracions..."

cat > configs/default.yaml << 'EOF'
model: unet_sr
in_channels: 7
out_channels: 2
n_filters: 64
scale_factor: 2
epochs: 50
batch_size: 4
learning_rate: 1.0e-4
train_split: 0.8
EOF

cat > scripts/quick_test.sh << 'SHEOF'
#!/bin/bash
set -e
cd /home/oriol/.openclaw/workspace/git/SRDownscalling
source /home/oriol/.openclaw/workspace/venv_sr_test/bin/activate
python -c "
import torch, sys
sys.path.insert(0, 'src')
from models import UNetSR
m = UNetSR()
x = torch.randn(1, 7, 50, 51)
y = m(x)
print(f'Input: {x.shape}, Output: {y.shape}')
print(f'Params: {sum(p.numel() for p in m.parameters()):,}')
"
SHEOF
chmod +x scripts/quick_test.sh

# ============================================================================
# 4. GIT I GITHUB
# ============================================================================
log "[4/6] Preparant GitHub..."

git config --global user.email "orioll@gmail.com"
git config --global user.name "Oriol"
git add -A 2>/dev/null || true
git commit -m "SRDownscalling: Complete setup with UNetSR and ESRGAN models" 2>/dev/null || log "No nous canvis"

log "Git preparat. Per pujar a GitHub executa:"
echo "  cd $REPO_DIR"
echo "  git remote add origin https://github.com/oriol/SRDownscalling.git"
echo "  git push -u origin main"

# ============================================================================
# 5. TEST RÀPID
# ============================================================================
log "[5/6] Executant test ràpid..."

python -c "
import torch, sys
sys.path.insert(0, 'src')
from models import UNetSR
m = UNetSR()
x = torch.randn(1, 7, 50, 51)
y = m(x)
print(f'✓ Model test: Input {x.shape} → Output {y.shape}')
print(f'✓ Parameters: {sum(p.numel() for p in m.parameters()):,}')
" 2>&1 | tee -a "$LOG_FILE"

# ============================================================================
# 6. DEPLOY A GPU SERVER
# ============================================================================
log "[6/6] Preparant deploy a GPU server..."

# Crear script de deploy
cat > scripts/deploy_to_gpu.sh << 'DEPLOYEOF'
#!/bin/bash
# Deploy to GPU server
GPU_SERVER="oriol@192.168.1.100"  # Canviar IP
GPU_DIR="/home/oriol/gpu_projects/SRDownscalling"

echo "Deploying to $GPU_SERVER..."
rsync -avz --exclude='outputs' --exclude='.git' /home/oriol/.openclaw/workspace/git/SRDownscalling/ "$GPU_SERVER:$GPU_DIR/"
echo "Done. Connect to GPU server and run: cd $GPU_DIR && bash scripts/train.sh"
DEPLOYEOF
chmod +x scripts/deploy_to_gpu.sh

cat > scripts/train.sh << 'TRAINEOF'
#!/bin/bash
# GPU Training Script
source ~/.venv/srdownscalling/bin/activate
cd /home/oriol/gpu_projects/SRDownscalling
python src/train.py --epochs 100 --batch_size 8 --lr 1e-4
TRAINEOF
chmod +x scripts/train.sh

log "========================================"
log "Setup complet!"
log "========================================"
log "Per pujar a GitHub:"
log "  cd $REPO_DIR"
log "  git remote add origin https://github.com/oriol/SRDownscalling.git"
log "  git push -u origin main"
log ""
log "Per fer deploy a GPU server:"
log "  bash $REPO_DIR/scripts/deploy_to_gpu.sh"
log ""
log "Log: $LOG_FILE"
