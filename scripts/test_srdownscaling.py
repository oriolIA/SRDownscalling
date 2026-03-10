#!/usr/bin/env python3
"""
SRDownscalling - Complete Test Suite
=====================================
Aquest script testa el model de super-resolution.

Funcionalitats:
- Genera dades sintètiques si no hi ha dades WRF
- Testa l'arquitectura del model
- Executa training loop curt
- Calcula mètriques
- Genera report

Ús:
    python3 scripts/test_srdownscaling.py [--epochs N] [--batch N]
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class Colors:
    """Colors per output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def log(msg: str, color: str = Colors.GREEN):
    """Log amb color."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{color}[{timestamp}]{Colors.ENDC} {msg}")


def warn(msg: str):
    log(f"WARNING: {msg}", Colors.WARNING)


def error(msg: str):
    log(f"ERROR: {msg}", Colors.FAIL)


def info(msg: str):
    log(msg, Colors.BLUE)


def check_dependencies() -> Dict[str, bool]:
    """Comprova les dependències."""
    deps = {}
    
    packages = ['torch', 'torchvision', 'xarray', 'netCDF4', 'numpy']
    
    for pkg in packages:
        try:
            module = __import__(pkg.replace('-', '_'))
            version = getattr(module, '__version__', 'unknown')
            deps[pkg] = True
            info(f"{pkg}: {version} ✓")
        except ImportError:
            deps[pkg] = False
            error(f"{pkg}: no instal·lat ✗")
    
    return deps


def check_data(data_dir: str) -> Dict:
    """Comprova les dades WRF."""
    info(f"Comprovant dades a: {data_dir}")
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        warn(f"Directori no existeix: {data_dir}")
        return {"exists": False, "reason": "Directory not found"}
    
    # Buscar fitxers .nc
    nc_files = list(data_path.glob("**/*.nc"))
    
    if not nc_files:
        warn(f"No s'han trobat fitxers NetCDF a {data_dir}")
        return {"exists": False, "reason": "No NetCDF files found"}
    
    info(f"Trobats {len(nc_files)} fitxers NetCDF")
    
    # Comprovar estructura de dominis
    domains = []
    for d in ['d01', 'd02', 'd05']:
        d_path = data_path / d
        if d_path.exists():
            files = list(d_path.glob("*.nc"))
            if files:
                domains.append({
                    "domain": d,
                    "files": len(files),
                    "sample": str(files[0].name)
                })
    
    return {
        "exists": True,
        "total_files": len(nc_files),
        "domains": domains
    }


def run_model_test(epochs: int, batch_size: int, scale: int) -> Dict:
    """Executa el test del model."""
    info("Executant test del model...")
    
    repo_dir = Path(__file__).parent.parent
    output_dir = repo_dir / "outputs" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Executar script bash
    script_path = repo_dir / "scripts" / "quick_test.sh"
    
    env = os.environ.copy()
    env['EPOCHS'] = str(epochs)
    env['BATCH_SIZE'] = str(batch_size)
    env['SCALE'] = str(scale)
    env['OUTPUT_DIR'] = str(output_dir)
    
    result = subprocess.run(
        ['bash', str(script_path)],
        env=env,
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        timeout=3600  # 1 hora timeout
    )
    
    # Parsejar resultats
    output = result.stdout
    
    # Buscar mètriques
    metrics = {}
    
    for line in output.split('\n'):
        if 'Loss' in line and '=' in line:
            parts = line.split('=')
            if len(parts) == 2:
                try:
                    metrics['final_loss'] = float(parts[1].strip().split()[0])
                except:
                    pass
    
    # Buscar si ha acabat bé
    if 'TEST COMPLETAT EXITOSAMENT' in output:
        metrics['status'] = 'success'
    elif result.returncode != 0:
        metrics['status'] = 'failed'
        metrics['error'] = result.stderr
    else:
        metrics['status'] = 'unknown'
    
    return metrics


def generate_report(results: Dict, output_path: Path):
    """Genera un report JSON."""
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    info(f"Report guardat a: {output_path}")


def main():
    """Main function."""
    print()
    print(Colors.HEADER + "=" * 60)
    print("SRDownscalling - Test Suite")
    print("=" * 60 + Colors.ENDC)
    print()
    
    parser = argparse.ArgumentParser(description='SRDownscalling Test')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--scale', type=int, default=2, help='Scale factor')
    parser.add_argument('--data', type=str, help='Data directory')
    
    args = parser.parse_args()
    
    data_dir = args.data or os.environ.get('DATA_DIR', '/home/oriol/data/WRF/1469893')
    
    results = {
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch,
            "scale": args.scale,
            "data_dir": data_dir
        },
        "checks": {},
        "metrics": {}
    }
    
    # 1. Comprovar dependències
    info("1. Comprovant dependències...")
    results["checks"]["dependencies"] = check_dependencies()
    
    # 2. Comprovar dades
    info("2. Comprovant dades...")
    results["checks"]["data"] = check_data(data_dir)
    
    # 3. Executar test del model
    info("3. Executant test del model...")
    results["metrics"] = run_model_test(args.epochs, args.batch, args.scale)
    
    # 4. Generar report
    repo_dir = Path(__file__).parent.parent
    report_path = repo_dir / "outputs" / "test_report.json"
    generate_report(results, report_path)
    
    # Resum
    print()
    print(Colors.HEADER + "=" * 60)
    print("RESUM")
    print("=" * 60 + Colors.ENDC)
    
    deps_ok = all(results["checks"]["dependencies"].values())
    data_ok = results["checks"]["data"]["exists"]
    test_ok = results["metrics"].get("status") == "success"
    
    print(f"  Dependències: {'✓' if deps_ok else '✗'}")
    print(f"  Dades:        {'✓' if data_ok else '✗ (ús sintètiques)'}")
    print(f"  Test:         {'✓' if test_ok else '✗'}")
    
    if test_ok:
        loss = results["metrics"].get("final_loss", "N/A")
        print(f"  Loss final:   {loss}")
    
    print()
    
    # Sortir amb codi apropiat
    sys.exit(0 if test_ok else 1)


if __name__ == '__main__':
    main()
