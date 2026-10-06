#!/bin/bash
set -e

echo "=== INFERNT Build Script ==="
echo ""

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. This script requires CUDA toolkit."
    echo "On Colab, CUDA should be pre-installed."
    exit 1
fi

echo "CUDA compiler:"
nvcc --version
echo ""

if command -v nvidia-smi &> /dev/null; then
    echo "GPU info:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building INFERNT..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
echo ""

echo "Build complete."
echo "Binary: $(pwd)/bin/infernt"
echo ""

echo "Running INFERNT Phase 1 tests..."
./bin/infernt
