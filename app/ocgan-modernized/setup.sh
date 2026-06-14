#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "=================================================="
echo "Ambiente creato correttamente in: $VENV_DIR"
python --version
pip --version
echo "=================================================="
echo "ATTENZIONE: PyTorch va installato a parte con il comando corretto per la CUDA della macchina."
echo "Esempio dopo attivazione venv:"
echo "source .venv/bin/activate"
echo "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
echo "=================================================="