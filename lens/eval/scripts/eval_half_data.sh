#!/usr/bin/env bash
# Narrative LLM-as-judge on a shared random subset of test indices (data-efficiency
# ablation: full vs half-data vs 10%-data checkpoints, all judged on the SAME 1000
# narrative indices). Runs ref-side once, then pred-side per candidate, merges, scores.
#
# Requirements: OPENAI_API_KEY set; the `eval` extra installed. No GPU.
# Env vars:
#   OUTPUT_DIR    output root (default ./llm_judge_results/eval_subset)
#   JUDGE_MODEL   judge model (default gpt-4.1-mini)
#   N_SAMPLE      subset size (default 1000)
#   SEED          sampling seed (default 2025)
#   INDEX_SOURCE  jsonl whose `index` field is sampled (default first candidate)
#
# Usage: positional args are "name=path/to/<candidate>_narrative_results.jsonl"
#   eval_half_data.sh \
#     "lens14b=inference_results/inference_results_LENS-14b/narrative_results.jsonl" \
#     "half=inference_results/inference_result_half_data/narrative_results.jsonl" \
#     "tenpct=inference_results/inference_result_10percent_data/narrative_results.jsonl"
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-./llm_judge_results/eval_subset}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-mini}"
N_SAMPLE="${N_SAMPLE:-1000}"
SEED="${SEED:-2025}"

[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "[ERROR] OPENAI_API_KEY is not set." >&2; exit 1; }
[[ "$#" -ge 1 ]] || { echo "usage: $0 name=path.jsonl ..." >&2; exit 2; }

mkdir -p "$OUTPUT_DIR/work"
declare -a NAMES=() PATHS=()
for spec in "$@"; do
  IFS='=' read -r n p <<< "$spec"
  NAMES+=("$n"); PATHS+=("$p")
done
INDEX_SOURCE="${INDEX_SOURCE:-${PATHS[0]}}"
INDEX_FILE="$OUTPUT_DIR/work/eval_indices_${N_SAMPLE}.txt"

echo "[0/4] sampling $N_SAMPLE indices from $INDEX_SOURCE (seed=$SEED)"
python - "$INDEX_SOURCE" "$INDEX_FILE" "$N_SAMPLE" "$SEED" <<'PY'
import json, random, sys
from pathlib import Path
src, out, n, seed = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
idxs = []
with src.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            o = json.loads(line)
            if "index" in o:
                idxs.append(str(o["index"]))
if len(idxs) < n:
    raise SystemExit(f"index count {len(idxs)} < n {n}")
out.write_text("\n".join(random.Random(seed).sample(idxs, n)), encoding="utf-8")
print(f"[OK] {n} indices -> {out}")
PY

echo "[1/4] ref-side judging (shared subset)"
python -m lens.eval.llm_judge.run_structured_eval \
  --input "$INDEX_SOURCE" --input-format jsonl \
  --reference-column reference --prediction-column prediction --max-rows -1 \
  --model "$JUDGE_MODEL" --side ref --index-file "$INDEX_FILE" \
  --output "$OUTPUT_DIR/work/ref_only.jsonl"

echo "[2/4] pred-side judging per candidate + [3/4] merge"
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"; path="${PATHS[$i]}"
  python -m lens.eval.llm_judge.run_structured_eval \
    --input "$path" --input-format jsonl \
    --reference-column reference --prediction-column prediction --max-rows -1 \
    --model "$JUDGE_MODEL" --side pred --index-file "$INDEX_FILE" \
    --output "$OUTPUT_DIR/work/${name}_pred_only.jsonl"
  python -m lens.eval.llm_judge.run_structured_eval --merge \
    "$OUTPUT_DIR/work/ref_only.jsonl" "$OUTPUT_DIR/work/${name}_pred_only.jsonl" \
    "$OUTPUT_DIR/${name}_merged_eval.jsonl"
done

echo "[4/4] metrics"
python -m lens.eval.metrics.compute_custom_metrics \
  --files "$OUTPUT_DIR"/*_merged_eval.jsonl \
  --out-csv "$OUTPUT_DIR/metrics_summary.csv" --out-json "$OUTPUT_DIR/metrics_full.json"
echo "[DONE] -> $OUTPUT_DIR"
