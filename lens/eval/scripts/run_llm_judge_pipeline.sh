#!/usr/bin/env bash
# LLM-as-a-Judge pipeline for a set of inference-result folders.
#
# For each folder that contains narrative_results.jsonl / qa_results.jsonl:
#   narrative: run_structured_eval (--side ref) + (--side pred) -> merge
#              -> compute_custom_metrics (Coverage / Presence / Severity Alignment)
#   qa:        run_qa_eval -> compute_qa_metrics (Severity Alignment)
#
# Requirements: OPENAI_API_KEY set; the `eval` extra installed (openai, pydantic).
# No GPU needed (judging is API-side). Uses the OpenAI Batch API (24h window).
#
# Env vars:
#   OUTPUT_ROOT       where per-folder eval artifacts go (default ./llm_judge_results)
#   JUDGE_MODEL       judge model (default gpt-4.1-mini)
#   POLL_INTERVAL     batch poll interval seconds (default 30)
#   COMPLETION_WINDOW OpenAI batch completion window (default 24h)
# Usage: run_llm_judge_pipeline.sh <inference_dir> [<inference_dir> ...]
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-./llm_judge_results}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-mini}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
COMPLETION_WINDOW="${COMPLETION_WINDOW:-24h}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] OPENAI_API_KEY is not set." >&2
  exit 1
fi
if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 <inference_dir> [<inference_dir> ...]" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
echo "[INFO] OUTPUT_ROOT=$OUTPUT_ROOT  JUDGE_MODEL=$JUDGE_MODEL"

for input_dir in "$@"; do
  [[ -d "$input_dir" ]] || { echo "[WARN] skip missing dir: $input_dir"; continue; }
  tag="$(basename "$input_dir")"
  work="$OUTPUT_ROOT/$tag"
  mkdir -p "$work"
  echo "============================================================"
  echo "[RUN] tag=$tag  input_dir=$input_dir"

  # ---------- narrative ----------
  narr=""
  for cand in "$input_dir/narrative_results.jsonl" "$input_dir/narrative_results_global_mismatch.jsonl"; do
    [[ -f "$cand" ]] && { narr="$cand"; break; }
  done
  if [[ -n "$narr" ]]; then
    for side in ref pred; do
      python -m lens.eval.llm_judge.run_structured_eval \
        --input "$narr" --input-format jsonl \
        --reference-column reference --prediction-column prediction --max-rows -1 \
        --model "$JUDGE_MODEL" --side "$side" \
        --output "$work/${tag}_narrative_${side}_eval.jsonl" \
        --completion-window "$COMPLETION_WINDOW" --poll-interval "$POLL_INTERVAL"
    done
    python -m lens.eval.llm_judge.run_structured_eval --merge \
      "$work/${tag}_narrative_ref_eval.jsonl" "$work/${tag}_narrative_pred_eval.jsonl" \
      "$work/${tag}_narrative_eval.jsonl"
    python -m lens.eval.metrics.compute_custom_metrics \
      --files "$work/${tag}_narrative_eval.jsonl" \
      --out-json "$work/${tag}_narrative_custom_metrics.json"
  else
    echo "[narrative] no narrative_results.jsonl found, skip"
  fi

  # ---------- qa ----------
  if [[ -f "$input_dir/qa_results.jsonl" ]]; then
    python -m lens.eval.llm_judge.run_qa_eval \
      --input "$input_dir/qa_results.jsonl" --input-format jsonl \
      --reference-column reference --prediction-column prediction --max-rows -1 \
      --model "$JUDGE_MODEL" --output "$work/${tag}_qa_eval.jsonl" \
      --completion-window "$COMPLETION_WINDOW" --poll-interval "$POLL_INTERVAL"
    python -m lens.eval.metrics.compute_qa_metrics \
      --files "$work/${tag}_qa_eval.jsonl" --output "$work/${tag}_qa_metrics.json"
  else
    echo "[qa] no qa_results.jsonl found, skip"
  fi
done

echo "[DONE] results under $OUTPUT_ROOT"
