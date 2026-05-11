#!/usr/bin/env bash
# LENS — Stage 1: time-series encoder alignment.
#
# Requires GPUs + the ChatTS-Training submodule (`./ChatTS-Training`, pinned at
# cf2139a8). NOT run by CI.
#
# Adapted from ChatTS-Training/scripts/train_chatts_stage1.sh — all absolute
# /workspace/... paths replaced by env vars with sensible defaults. Override
# any of them in the environment before calling this script:
#
#   LENS_REPO_ROOT     repo root (auto-detected from this script's location)
#   LENS_CHATTS_DIR    the ChatTS-Training submodule (default: $LENS_REPO_ROOT/ChatTS-Training)
#   LENS_MODEL_DIR     base model path / HF id            (default: Qwen/Qwen2.5-14B)
#   LENS_DATA_DIR      dataset_dir passed to LLaMA-Factory (default: data, relative to $LENS_CHATTS_DIR)
#   LENS_OUTPUT_DIR    where checkpoints land             (default: $LENS_REPO_ROOT/models)
#   LENS_DEEPSPEED     DeepSpeed config (default: $LENS_REPO_ROOT/configs/training/deepspeed/ds_config_2.json)
#   LENS_NUM_GPUS      GPUs per node                      (default: 8)
#   LENS_MASTER_PORT   deepspeed launcher port            (default: 19901)
#   LENS_STAGE_YAML    training YAML template             (default: $LENS_REPO_ROOT/configs/training/stage1.yaml)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LENS_REPO_ROOT="${LENS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

LENS_CHATTS_DIR="${LENS_CHATTS_DIR:-$LENS_REPO_ROOT/ChatTS-Training}"
LENS_MODEL_DIR="${LENS_MODEL_DIR:-Qwen/Qwen2.5-14B}"
LENS_DATA_DIR="${LENS_DATA_DIR:-data}"
LENS_OUTPUT_DIR="${LENS_OUTPUT_DIR:-$LENS_REPO_ROOT/models}"
LENS_DEEPSPEED="${LENS_DEEPSPEED:-$LENS_REPO_ROOT/configs/training/deepspeed/ds_config_2.json}"
LENS_NUM_GPUS="${LENS_NUM_GPUS:-8}"
LENS_MASTER_PORT="${LENS_MASTER_PORT:-19901}"
LENS_STAGE_YAML="${LENS_STAGE_YAML:-$LENS_REPO_ROOT/configs/training/stage1.yaml}"

# Optional CUDA env (only exported if CUDA_HOME is already set / present).
if [[ -d "${CUDA_HOME:-/usr/local/cuda}" ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
export DISABLE_VERSION_CHECK=1

if [[ ! -d "$LENS_CHATTS_DIR/src" ]]; then
  echo "error: ChatTS-Training not found at $LENS_CHATTS_DIR" >&2
  echo "       run: git submodule update --init ChatTS-Training   (or git clone --recursive)" >&2
  exit 1
fi

# Render the YAML template (${VAR} placeholders -> concrete paths).
RENDERED_YAML="$(mktemp -t lens_stage1_XXXXXX.yaml)"
trap 'rm -f "$RENDERED_YAML"' EXIT
export LENS_MODEL_DIR LENS_DATA_DIR LENS_OUTPUT_DIR LENS_DEEPSPEED
envsubst '${LENS_MODEL_DIR} ${LENS_DATA_DIR} ${LENS_OUTPUT_DIR} ${LENS_DEEPSPEED}' \
  < "$LENS_STAGE_YAML" > "$RENDERED_YAML"

echo "LENS Stage 1 — encoder alignment"
echo "  repo root : $LENS_REPO_ROOT"
echo "  chatts    : $LENS_CHATTS_DIR"
echo "  model     : $LENS_MODEL_DIR"
echo "  data dir  : $LENS_DATA_DIR"
echo "  output    : $LENS_OUTPUT_DIR/lens-stage1"
echo "  deepspeed : $LENS_DEEPSPEED"
echo "  gpus      : $LENS_NUM_GPUS"
echo "  yaml      : $LENS_STAGE_YAML -> $RENDERED_YAML"

cd "$LENS_CHATTS_DIR"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}" DEEPSPEED_TIMEOUT="${DEEPSPEED_TIMEOUT:-120}" \
  deepspeed --num_gpus "$LENS_NUM_GPUS" --master_port "$LENS_MASTER_PORT" \
    src/train.py "$RENDERED_YAML"
