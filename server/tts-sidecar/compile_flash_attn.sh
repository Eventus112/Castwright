#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=4

echo "=== 1. Checking environment ==="
./server/tts-sidecar/.venv/bin/python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"
which nvcc || true
nvcc --version || true

echo "=== 2. Installing ninja ==="
./server/tts-sidecar/.venv/bin/pip install ninja

echo "=== 3. Compiling flash-attn ==="
./server/tts-sidecar/.venv/bin/pip install flash-attn --no-build-isolation --no-cache-dir

echo "=== 4. Testing import ==="
./server/tts-sidecar/.venv/bin/python -c "import flash_attn; print('SUCCESS: flash_attn', flash_attn.__version__)"
