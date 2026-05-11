#!/usr/bin/env bash
# Run HF inference (narrative + qa test splits) over a directory of ablation
# checkpoints (e.g. encoder patch-size / MLP-depth variants).
#
# Hardware: GPUs required. Launches lens.eval.inference.hf_inference via accelerate.
# Env vars:
#   MODEL_ROOT          directory whose immediate subdirs are checkpoints (required)
#   OUTPUT_ROOT         output root (default ./inference_results_ablation)
#   DATA_ROOT           dataset root (default data/fake)
#   DATASETS            space-separated jsonl paths rel. to DATA_ROOT
#                       (default "narrative_dataset/test.jsonl qa_dataset/test.jsonl")
#   CUDA_VISIBLE_DEVICES, NUM_PROCESSES, MAX_SAMPLES (-1=all), BATCH_SIZE
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:?set MODEL_ROOT to a directory of checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./inference_results_ablation}"
DATA_ROOT="${DATA_ROOT:-data/fake}"
DATASETS="${DATASETS:-narrative_dataset/test.jsonl qa_dataset/test.jsonl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
BATCH_SIZE="${BATCH_SIZE:-1}"

[[ -d "$MODEL_ROOT" ]] || { echo "[ERROR] MODEL_ROOT not found: $MODEL_ROOT" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"
echo "[INFO] MODEL_ROOT=$MODEL_ROOT  OUTPUT_ROOT=$OUTPUT_ROOT  GPUs=$CUDA_VISIBLE_DEVICES"

shopt -s nullglob
for model_dir in "$MODEL_ROOT"/*/; do
  name="$(basename "$model_dir")"
  out_dir="$OUTPUT_ROOT/$name"
  mkdir -p "$out_dir"
  echo "------------------------------------------------------------"
  echo "[RUN] $name"
  # shellcheck disable=SC2086
  accelerate launch --num_processes "$NUM_PROCESSES" -m lens.eval.inference.hf_inference \
    --model-path "$model_dir" --data-root "$DATA_ROOT" --datasets $DATASETS \
    --output-dir "$out_dir" --max-samples-per-dataset "$MAX_SAMPLES" --batch-size "$BATCH_SIZE"
done
echo "[DONE] -> $OUTPUT_ROOT"
