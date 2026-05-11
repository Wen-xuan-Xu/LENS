#!/usr/bin/env bash
# Start an SGLang server for the TS-Image baseline VLM (Qwen2.5-VL 7B / 32B).
#
# NOTE: query_vlm.py uses the SGLang *offline Engine* (it loads the model itself),
# so you usually do NOT need a separate server. This launcher is provided for the
# server-mode workflow / debugging. Hardware: GPUs required.
#
# Env vars: MODEL_PATH (or $VLM_MODEL_DIR), PORT (30000), HOST (0.0.0.0),
#   TP_SIZE (4), MEM_FRACTION_STATIC (0.7), SIF_IMAGE, WORKSPACE_DIR,
#   CUDA_VISIBLE_DEVICES, SINGULARITY_CACHEDIR, NO_CONTAINER=1
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-${VLM_MODEL_DIR:?set MODEL_PATH or VLM_MODEL_DIR}}"
PORT="${PORT:-30000}"; HOST="${HOST:-0.0.0.0}"
TP_SIZE="${TP_SIZE:-4}"; MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.7}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "==================== SGLang (TS-Image VLM) ===================="
echo "model=$MODEL_PATH  host=$HOST  port=$PORT  TP=$TP_SIZE  GPUs=$CUDA_VISIBLE_DEVICES"

if [[ "${NO_CONTAINER:-0}" == "1" ]]; then
  exec python3 -m sglang.launch_server --model-path "$MODEL_PATH" \
    --port "$PORT" --host "$HOST" --tp-size "$TP_SIZE" --mem-fraction-static "$MEM_FRACTION_STATIC"
fi

SIF_IMAGE="${SIF_IMAGE:-./sglang_latest.sif}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$PWD}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-$HOME/.singularity}"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
if [[ ! -f "$SIF_IMAGE" ]]; then
  echo "[INFO] building SGLang SIF image -> $SIF_IMAGE"
  mkdir -p "$(dirname "$SIF_IMAGE")"
  singularity build "$SIF_IMAGE" docker://lmsysorg/sglang:latest
fi
exec singularity run --nv \
  --bind "${WORKSPACE_DIR}:/workspace:rw" \
  --env TRITON_CACHE_DIR=/tmp/triton_cache \
  "$SIF_IMAGE" \
  python3 -m sglang.launch_server --model-path "$MODEL_PATH" \
    --port "$PORT" --host "$HOST" --tp-size "$TP_SIZE" --mem-fraction-static "$MEM_FRACTION_STATIC"
