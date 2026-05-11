#!/usr/bin/env bash
# Run the QA LLM-as-judge over several inference-result files in one go.
#
# Requirements: OPENAI_API_KEY set; the `eval` extra installed. No GPU.
# Env vars:
#   OUTPUT_DIR   where the *_qa_eval.jsonl files go (default ./llm_judge_results)
#   JUDGE_MODEL  judge model (default gpt-4.1-mini)
#
# Usage: each positional arg is "name:format:input_path:ref_col:pred_col[:question_col]"
#   run_qa_eval_batch.sh \
#     "lens14b:jsonl:inference_results/inference_results_LENS-14b/qa_results.jsonl:reference:prediction" \
#     "ts_image:csv:lens/baselines/ts_image/out/qa_vlm_responses.csv:ground_truth:vlm_response:question"
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-./llm_judge_results}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-mini}"

[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "[ERROR] OPENAI_API_KEY is not set." >&2; exit 1; }
[[ "$#" -ge 1 ]] || { echo "usage: $0 name:fmt:path:ref_col:pred_col[:q_col] ..." >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"

for spec in "$@"; do
  IFS=':' read -r name fmt path ref_col pred_col q_col <<< "$spec"
  echo "=============================================="
  echo "[QA] $name  ($fmt)  $path"
  args=( --input "$path" --input-format "$fmt"
         --reference-column "$ref_col" --prediction-column "$pred_col"
         --max-rows -1 --model "$JUDGE_MODEL"
         --output "$OUTPUT_DIR/qa_${name}_eval.jsonl" )
  [[ -n "${q_col:-}" ]] && args+=( --question-column "$q_col" )
  python -m lens.eval.llm_judge.run_qa_eval "${args[@]}"
done

echo "=============================================="
echo "[DONE] -> $OUTPUT_DIR ; now run:"
echo "  python -m lens.eval.metrics.compute_qa_metrics --files $OUTPUT_DIR/qa_*_eval.jsonl"
