#!/usr/bin/env bash
# Stage-2 SFT ablation suite for the LENS time-series encoder: re-run the stage-2
# recipe while varying only the encoder hyper-parameters in the base model's
# config.json (ts.patch_size and ts.num_layers). Produces one checkpoint per
# variant; pair with run_ablation_hf_inference_suite.sh + the LLM-judge pipeline.
#
# This script delegates training to the ChatTS-Training submodule
# (./ChatTS-Training); set CHATTS_TRAINING_DIR to override. See lens/training/
# for the stage-1/stage-2 launchers.
#
# Hardware: multi-GPU. Env vars:
#   CHATTS_TRAINING_DIR  path to the ChatTS-Training checkout (required)
#   BASE_MODEL_DIR       base Qwen2.5-14B (or $MODEL_DIR) with a `ts` block in config.json
#   VARIANT_ROOT         where per-variant model dirs are assembled (required)
#   OUTPUT_ROOT          where per-variant checkpoints go (required)
#   DATA_ROOT            dataset root (default data/fake)
#   CUDA_VISIBLE_DEVICES, NUM_GPUS, MASTER_PORT
#   MAX_STEPS, PER_DEVICE_BS, GRAD_ACC, LEARNING_RATE, WARMUP_RATIO, CUTOFF_LEN
#
# Variants run: ablate_patch_p4_l5, ablate_patch_p16_l5, ablate_layers_p8_l3,
#               ablate_layers_p8_l7 (baseline p8/l5 assumed already trained;
#               set RUN_BASELINE=1 to also (re)train it).
set -euo pipefail

CHATTS_TRAINING_DIR="${CHATTS_TRAINING_DIR:?set CHATTS_TRAINING_DIR}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${MODEL_DIR:?set BASE_MODEL_DIR or MODEL_DIR}}"
VARIANT_ROOT="${VARIANT_ROOT:?set VARIANT_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
DATA_ROOT="${DATA_ROOT:-data/fake}"
RUN_BASELINE="${RUN_BASELINE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-19921}"
MAX_STEPS="${MAX_STEPS:-40}"
PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
GRAD_ACC="${GRAD_ACC:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.02}"
CUTOFF_LEN="${CUTOFF_LEN:-4000}"

mkdir -p "$VARIANT_ROOT" "$OUTPUT_ROOT"
cd "$CHATTS_TRAINING_DIR"

assemble_variant() {  # variant_dir patch_size num_layers
  local vdir="$1" ps="$2" nl="$3"
  mkdir -p "$vdir"
  shopt -s nullglob
  for src in "$BASE_MODEL_DIR"/*; do
    local name; name="$(basename "$src")"
    [[ "$name" == "config.json" ]] && continue
    [[ -e "$vdir/$name" ]] || ln -s "$src" "$vdir/$name"
  done
  shopt -u nullglob
  cp "$BASE_MODEL_DIR/config.json" "$vdir/config.json"
  python - "$vdir/config.json" "$ps" "$nl" <<'PY'
import json, sys
from pathlib import Path
cfg_path, ps, nl = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
if "ts" not in cfg or not isinstance(cfg["ts"], dict):
    raise SystemExit(f"{cfg_path} has no `ts` block")
cfg["ts"]["patch_size"] = ps
cfg["ts"]["num_layers"] = nl
cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[OK] {cfg_path}: patch_size={ps}, num_layers={nl}")
PY
}

run_one() {  # exp_name patch_size num_layers
  local name="$1" ps="$2" nl="$3"
  local mdir="$VARIANT_ROOT/$name" odir="$OUTPUT_ROOT/$name"
  mkdir -p "$odir"
  echo "============================================================"
  echo "[RUN] $name  patch_size=$ps  num_layers=$nl"
  assemble_variant "$mdir" "$ps" "$nl"
  NCCL_DEBUG=WARN deepspeed --master_port "$MASTER_PORT" src/train.py \
    --deepspeed ds_config/ds_config_3.json \
    --stage sft --model_name_or_path "$mdir" \
    --dataset "narrative_dataset,qa_dataset,ift,align_random" \
    --interleave_probs "0.3,0.3,0.2,0.2" --do_train --mix_strategy "interleave_over" \
    --template "chatts" --finetuning_type full \
    --dataset_dir "$DATA_ROOT" \
    --output_dir "$odir" --overwrite_output_dir \
    --per_device_train_batch_size "$PER_DEVICE_BS" --gradient_accumulation_steps "$GRAD_ACC" \
    --lr_scheduler_type cosine --logging_steps 1 --save_steps 200 \
    --learning_rate "$LEARNING_RATE" --warmup_ratio "$WARMUP_RATIO" \
    --num_train_epochs 0 --max-steps "$MAX_STEPS" --plot_loss --bf16 \
    --save_only_model --save_safetensors False --preprocessing_num_workers 16 \
    --trust_remote_code True --cutoff_len "$CUTOFF_LEN"
}

[[ "$RUN_BASELINE" == "1" ]] && run_one "baseline_p8_l5" 8 5 || echo "[SKIP] baseline_p8_l5"
run_one "ablate_patch_p4_l5"  4 5
run_one "ablate_patch_p16_l5" 16 5
run_one "ablate_layers_p8_l3" 8 3
run_one "ablate_layers_p8_l7" 8 7
echo "[DONE] -> $OUTPUT_ROOT"
