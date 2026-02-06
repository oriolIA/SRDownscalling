#!/usr/bin/env python3
"""
SRDownscalling - Standalone Test Script
=========================================
Aquest script funciona sense dependre del sistema.
Genera dades sintètiques i testa el model.

Ús:
    python3 scripts/run_test.py [--epochs N] [--batch N] [--output DIR]
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def run_test(epochs=1, batch_size=4, scale=2, output_dir=None):
    """Executa el test del model."""
    
    repo_dir = Path(__file__).parent.parent
    if output_dir is None:
        output_dir = repo_dir / "outputs" / "quick_test"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar entorn
    env = os.environ.copy()
    env['OUTPUT_DIR'] = str(output_dir)
    env['EPOCHS'] = str(epochs)
    env['BATCH_SIZE'] = str(batch_size)
    env['SCALE'] = str(scale)
    env['DATA_DIR'] = '/home/oriol/data/WRF/1469893'
    
    # Executar script bash
    script_path = repo_dir / "scripts" / "quick_test.sh"
    
    print(f"Executant test...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch: {batch_size}")
    print(f"  Scale: {scale}")
    print(f"  Output: {output_dir}")
    
    result = subprocess.run(
        ['bash', str(script_path)],
        env=env,
        cwd=str(repo_dir),
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='SRDownscalling Test')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--scale', type=int, default=2, help='Scale factor')
    parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    success = run_test(
        epochs=args.epochs,
        batch_size=args.batch,
        scale=args.scale,
        output_dir=args.output
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
