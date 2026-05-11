#!/usr/bin/env bash
# Start an SGLang OpenAI-compatible server for the TS-Text baseline model.
#
# Hardware: GPUs required (tensor-parallel over $TP_SIZE GPUs). Runs inside a
# Singularity/Apptainer container by default (SIF image from docker://lmsysorg/sglang).
#
# Env vars (all optional):
#   MODEL_PATH        path/HF-id of the baseline model (default: $MODEL_DIR)
#   PORT, HOST        server bind (default 30000 / 0.0.0.0)
#   TP_SIZE, DP_SIZE  tensor / data parallel sizes (default 4 / 1)
#   SIF_IMAGE         path to the SGLang .sif (built on first use if missing)
#   WORKSPACE_DIR     host dir bind-mounted to /workspace (default $PWD)
#   CUDA_VISIBLE_DEVICES, SINGULARITY_CACHEDIR
#   MEM_FRACTION_STATIC  (default 0.9)
#   NO_CONTAINER=1    run sglang directly on the host instead of via Singularity
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-${MODEL_DIR:?set MODEL_PATH or MODEL_DIR}}"
PORT="${PORT:-30000}"; HOST="${HOST:-0.0.0.0}"
TP_SIZE="${TP_SIZE:-4}"; DP_SIZE="${DP_SIZE:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.9}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "==================== SGLang (TS-Text baseline) ===================="
echo "model=$MODEL_PATH  host=$HOST  port=$PORT  TP=$TP_SIZE  DP=$DP_SIZE  GPUs=$CUDA_VISIBLE_DEVICES"

if [[ "${NO_CONTAINER:-0}" == "1" ]]; then
  exec python3 -m sglang.launch_server --model-path "$MODEL_PATH" \
    --port "$PORT" --host "$HOST" --tp-size "$TP_SIZE" --dp-size "$DP_SIZE" \
    --mem-fraction-static "$MEM_FRACTION_STATIC"
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
    --port "$PORT" --host "$HOST" --tp-size "$TP_SIZE" --dp-size "$DP_SIZE" \
    --mem-fraction-static "$MEM_FRACTION_STATIC"
