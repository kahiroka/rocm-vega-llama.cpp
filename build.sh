#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-rocm72-vega-gfx900-gfx906}"
LLAMA_DIR="${LLAMA_DIR:-$HOME/Development/llama.cpp}"
BUILD_DIR_NAME="${BUILD_DIR_NAME:-build-rocm72-dual}"
JOBS="${JOBS:-$(nproc)}"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed or not in PATH" >&2
  exit 1
fi

if [[ ! -f Dockerfile.rocm72-gfx900-gfx906 ]]; then
  echo "ERROR: run this script from the repository root" >&2
  exit 1
fi

if [[ ! -f "$LLAMA_DIR/CMakeLists.txt" ]]; then
  echo "ERROR: llama.cpp source not found at: $LLAMA_DIR" >&2
  echo "Set LLAMA_DIR to your llama.cpp checkout." >&2
  exit 1
fi

echo "==> Building mixed ROCm image: $IMAGE_NAME"
sudo docker build \
  -t "$IMAGE_NAME" \
  -f Dockerfile.rocm72-gfx900-gfx906 \
  .

echo "==> Verifying gfx900 and gfx906 rocBLAS payloads"
sudo docker run --rm "$IMAGE_NAME" bash -lc '
  set -e
  LIB=/opt/rocm/lib/rocblas/library
  GFX900=$(find "$LIB" -maxdepth 1 -type f | grep -c gfx900 || true)
  GFX906=$(find "$LIB" -maxdepth 1 -type f | grep -c gfx906 || true)
  echo "gfx900 files: $GFX900"
  echo "gfx906 files: $GFX906"
  test "$GFX900" -gt 0
  test "$GFX906" -gt 0
'

echo "==> Building llama-server for gfx900 + gfx906"
sudo docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  -v "$LLAMA_DIR:/workspace/llama.cpp" \
  "$IMAGE_NAME" \
  bash -lc "
    set -euo pipefail
    cd /workspace/llama.cpp
    cmake -B '$BUILD_DIR_NAME' \\
      -DGGML_HIP=ON \\
      -DGPU_TARGETS='gfx900;gfx906' \\
      -DCMAKE_BUILD_TYPE=Release \\
      -DLLAMA_BUILD_SERVER=ON
    cmake --build '$BUILD_DIR_NAME' \\
      --target llama-server \\
      -j'$JOBS'
    ./'$BUILD_DIR_NAME'/bin/llama-server --list-devices
  "

echo
echo "Build complete."
echo "Persistent binary: $LLAMA_DIR/$BUILD_DIR_NAME/bin/llama-server"
echo "Image: $IMAGE_NAME"
