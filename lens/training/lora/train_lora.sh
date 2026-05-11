#!/usr/bin/env bash
# LENS — LoRA fine-tuning variant.
#
# Requires GPUs + the ChatTS-Training submodule (`./ChatTS-Training`, pinned at
# cf2139a8). NOT run by CI.
#
# Adapted from LENS/Sympt2Text/.../Exp4_2/core/train/run_lora_train.sh +
# ds_config_lora.json (which drove a bespoke torchrun trainer). Here the LoRA
# run goes through the ChatTS-Training / LLaMA-Factory LoRA path so it reuses
# the same template, datasets and DeepSpeed plumbing as the full SFT stages.
#
# Env vars (all optional; sensible defaults):
#   LENS_REPO_ROOT     repo root (auto-detected from this script's location)
#   LENS_CHATTS_DIR    the ChatTS-Training submodule (default: $LENS_REPO_ROOT/ChatTS-Training)
#   LENS_MODEL_DIR     base model path / HF id   (default: Qwen/Qwen2.5-14B)
#   LENS_DATA_DIR      dataset_dir for LLaMA-Factory (default: data, relative to $LENS_CHATTS_DIR)
#   LENS_OUTPUT_DIR    where adapters land       (default: $LENS_REPO_ROOT/models)
#   LENS_DEEPSPEED     DeepSpeed config          (default: $LENS_REPO_ROOT/lens/training/lora/ds_config_lora.json)
#   LENS_NUM_GPUS      GPUs per node             (default: 4)
#   LENS_MASTER_PORT   deepspeed launcher port   (default: 19902)
#   LENS_STAGE_YAML    training YAML template    (default: $LENS_REPO_ROOT/configs/training/lora.yaml)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LENS_REPO_ROOT="${LENS_REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

LENS_CHATTS_DIR="${LENS_CHATTS_DIR:-$LENS_REPO_ROOT/ChatTS-Training}"
LENS_MODEL_DIR="${LENS_MODEL_DIR:-Qwen/Qwen2.5-14B}"
LENS_DATA_DIR="${LENS_DATA_DIR:-data}"
LENS_OUTPUT_DIR="${LENS_OUTPUT_DIR:-$LENS_REPO_ROOT/models}"
LENS_DEEPSPEED="${LENS_DEEPSPEED:-$LENS_REPO_ROOT/lens/training/lora/ds_config_lora.json}"
LENS_NUM_GPUS="${LENS_NUM_GPUS:-4}"
LENS_MASTER_PORT="${LENS_MASTER_PORT:-19902}"
LENS_STAGE_YAML="${LENS_STAGE_YAML:-$LENS_REPO_ROOT/configs/training/lora.yaml}"

if [[ -d "${CUDA_HOME:-/usr/local/cuda}" ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
export DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ ! -d "$LENS_CHATTS_DIR/src" ]]; then
  echo "error: ChatTS-Training not found at $LENS_CHATTS_DIR" >&2
  echo "       run: git submodule update --init ChatTS-Training   (or git clone --recursive)" >&2
  exit 1
fi

RENDERED_YAML="$(mktemp -t lens_lora_XXXXXX.yaml)"
trap 'rm -f "$RENDERED_YAML"' EXIT
export LENS_MODEL_DIR LENS_DATA_DIR LENS_OUTPUT_DIR LENS_DEEPSPEED
envsubst '${LENS_MODEL_DIR} ${LENS_DATA_DIR} ${LENS_OUTPUT_DIR} ${LENS_DEEPSPEED}' \
  < "$LENS_STAGE_YAML" > "$RENDERED_YAML"

echo "LENS — LoRA fine-tuning"
echo "  repo root : $LENS_REPO_ROOT"
echo "  chatts    : $LENS_CHATTS_DIR"
echo "  model     : $LENS_MODEL_DIR"
echo "  data dir  : $LENS_DATA_DIR"
echo "  output    : $LENS_OUTPUT_DIR/lens-lora"
echo "  deepspeed : $LENS_DEEPSPEED"
echo "  gpus      : $LENS_NUM_GPUS"
echo "  yaml      : $LENS_STAGE_YAML -> $RENDERED_YAML"

cd "$LENS_CHATTS_DIR"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}" DEEPSPEED_TIMEOUT="${DEEPSPEED_TIMEOUT:-120}" \
  deepspeed --num_gpus "$LENS_NUM_GPUS" --master_port "$LENS_MASTER_PORT" \
    src/train.py "$RENDERED_YAML"
