#!/usr/bin/env bash
# Train the TS-Text-FT baseline (text-only fine-tune of Qwen2.5-14B) via LLaMA-Factory.
#
# Hardware: multi-GPU. Requires a LLaMA-Factory install (pip or a checkout) with
# the LENS datasets registered in its data/dataset_info.json (see README.md). This
# baseline is intentionally near-stock LLaMA-Factory — we vendor only the YAML.
#
# Env vars:
#   MODEL_DIR    base model path / HF id (e.g. Qwen/Qwen2.5-14B)   [required]
#   DATA_ROOT    dataset_dir holding dataset_info.json + the dataset folders [required]
#   OUTPUT_DIR   where the checkpoint is written                   [required]
#   CONFIG_FILE  training YAML (default: the qwen2_5_14b_text_ft.yaml next to this script)
#   LLAMAFACTORY_DIR  optional: cd into a LLaMA-Factory checkout first
#   MASTER_PORT  torchrun master port (default 19901)
#   CUDA_VISIBLE_DEVICES
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR (base model path / HF id)}"
: "${DATA_ROOT:?set DATA_ROOT (dataset dir with dataset_info.json)}"
: "${OUTPUT_DIR:?set OUTPUT_DIR (checkpoint output dir)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/qwen2_5_14b_text_ft.yaml}"
export MASTER_PORT="${MASTER_PORT:-19901}"
export DISABLE_VERSION_CHECK=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [[ -n "${LLAMAFACTORY_DIR:-}" ]]; then
  cd "$LLAMAFACTORY_DIR"
fi

# Substitute ${MODEL_DIR}/${DATA_ROOT}/${OUTPUT_DIR} into a temp config.
RESOLVED_CONFIG="$(mktemp --suffix=.yaml)"
trap 'rm -f "$RESOLVED_CONFIG"' EXIT
MODEL_DIR="$MODEL_DIR" DATA_ROOT="$DATA_ROOT" OUTPUT_DIR="$OUTPUT_DIR" \
  envsubst '${MODEL_DIR} ${DATA_ROOT} ${OUTPUT_DIR}' < "$CONFIG_FILE" > "$RESOLVED_CONFIG"

echo "[INFO] config=$CONFIG_FILE  MODEL_DIR=$MODEL_DIR  DATA_ROOT=$DATA_ROOT  OUTPUT_DIR=$OUTPUT_DIR  MASTER_PORT=$MASTER_PORT"
llamafactory-cli train "$RESOLVED_CONFIG"
echo "[DONE] checkpoint -> $OUTPUT_DIR"
