#!/usr/bin/env bash
# Run the TS-Image VLM query (lens.baselines.ts_image.query_vlm) inside a
# Singularity/Apptainer container (the SGLang image bundles vLLM/FlashInfer etc.).
#
# Hardware: GPUs required. Caches (FlashInfer / Triton) are redirected to writable
# scratch dirs to avoid home-directory quota issues.
#
# Env vars: VLM_MODEL_DIR (required), DATA_ROOT (data/fake), SIF_IMAGE,
#   WORKSPACE_DIR (host dir bind-mounted to /workspace; default $PWD),
#   FLASHINFER_CACHE_HOST (default ./_flashinfer_cache), CUDA_VISIBLE_DEVICES,
#   SINGULARITY_CACHEDIR
# Any extra args are forwarded to query_vlm.py (e.g. --model-size 32b --dataset-type qa).
set -euo pipefail

: "${VLM_MODEL_DIR:?set VLM_MODEL_DIR}"
DATA_ROOT="${DATA_ROOT:-data/fake}"
SIF_IMAGE="${SIF_IMAGE:-./sglang_latest.sif}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$PWD}"
FLASHINFER_CACHE_HOST="${FLASHINFER_CACHE_HOST:-./_flashinfer_cache}"
FLASHINFER_CACHE_CONTAINER="/workspace/.cache/flashinfer"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-$HOME/.singularity}"
export APPTAINER_CACHEDIR="$SINGULARITY_CACHEDIR"
mkdir -p "$FLASHINFER_CACHE_HOST"

if [[ ! -f "$SIF_IMAGE" ]]; then
  echo "[INFO] building SGLang SIF image -> $SIF_IMAGE"
  mkdir -p "$(dirname "$SIF_IMAGE")"
  singularity build "$SIF_IMAGE" docker://lmsysorg/sglang:latest
fi

echo "==================== TS-Image VLM query (container) ===================="
echo "model=$VLM_MODEL_DIR  data_root=$DATA_ROOT  GPUs=$CUDA_VISIBLE_DEVICES"

singularity exec --nv \
  --bind "${WORKSPACE_DIR}:/workspace:rw" \
  --bind "$FLASHINFER_CACHE_HOST:$FLASHINFER_CACHE_CONTAINER:rw" \
  --env TRITON_CACHE_DIR=/tmp/triton_cache \
  --env FLASHINFER_WORKSPACE_DIR="$FLASHINFER_CACHE_CONTAINER" \
  --env FLASHINFER_CACHE_DIR="$FLASHINFER_CACHE_CONTAINER" \
  --env XDG_CACHE_HOME="$FLASHINFER_CACHE_CONTAINER" \
  --env DATA_ROOT="$DATA_ROOT" \
  --env VLM_MODEL_DIR="$VLM_MODEL_DIR" \
  "$SIF_IMAGE" \
  python3 -m lens.baselines.ts_image.query_vlm "$@"

echo "[DONE]"
