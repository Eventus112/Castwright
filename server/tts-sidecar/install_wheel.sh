#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

echo "=== 1. Downloading matching cxx11abiFALSE wheel ==="
curl -L "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl" -o ./server/tts-sidecar/flash_attn_cxx11abiFALSE.whl

echo "=== 2. Installing wheel ==="
./server/tts-sidecar/.venv/bin/pip install ./server/tts-sidecar/flash_attn_cxx11abiFALSE.whl --force-reinstall --no-deps

echo "=== 3. Verifying import ==="
./server/tts-sidecar/.venv/bin/python -c "import flash_attn; print('SUCCESS! flash_attn version:', flash_attn.__version__)"
