#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-rocm72-vega-gfx900-gfx906}"
LLAMA_DIR="${LLAMA_DIR:-$HOME/Development/llama.cpp}"
BUILD_DIR_NAME="${BUILD_DIR_NAME:-build-rocm72-dual}"
MODEL_DIR="${MODEL_DIR:-$HOME/Downloads}"
MODEL="${MODEL:-gemma-4-26B-A4B-it-UD-Q4_K_M.gguf}"
CONTEXT="${CONTEXT:-8192}"
TENSOR_SPLIT="${TENSOR_SPLIT:-1,4}"
PORT="${PORT:-8080}"

SERVER="$LLAMA_DIR/$BUILD_DIR_NAME/bin/llama-server"
MODEL_PATH="$MODEL_DIR/$MODEL"

if [[ ! -x "$SERVER" ]]; then
  echo "ERROR: llama-server not found at: $SERVER" >&2
  echo "Run ./build.sh first, or set LLAMA_DIR/BUILD_DIR_NAME." >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: model not found at: $MODEL_PATH" >&2
  echo "Set MODEL_DIR and MODEL to the GGUF you want to run." >&2
  exit 1
fi

echo "==> Mixed Vega llama.cpp server"
echo "    model:        $MODEL_PATH"
echo "    context:      $CONTEXT"
echo "    tensor split: $TENSOR_SPLIT"
echo "    port:         $PORT"
echo

sudo docker run --rm -it \
  -p "$PORT:$PORT" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  -v "$LLAMA_DIR:/workspace/llama.cpp" \
  -v "$MODEL_DIR:/models:ro" \
  "$IMAGE_NAME" \
  bash -lc "
    set -e
    cd /workspace/llama.cpp

    echo '==> Visible llama.cpp devices'
    ./'$BUILD_DIR_NAME'/bin/llama-server --list-devices
    echo

    exec ./'$BUILD_DIR_NAME'/bin/llama-server \\
      -m '/models/$MODEL' \\
      --device ROCm0,ROCm1 \\
      -ngl 999 \\
      --split-mode layer \\
      --tensor-split '$TENSOR_SPLIT' \\
      -c '$CONTEXT' \\
      -np 1 \\
      -b 512 \\
      -ub 256 \\
      --host 0.0.0.0 \\
      --port '$PORT'
  "
