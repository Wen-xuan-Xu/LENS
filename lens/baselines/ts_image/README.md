# `ts_image` — chart-image few-shot baseline

The **TS-Image** baseline renders the seven raw passive-sensing streams as a
multi-panel chart image and feeds it (plus the sleep/conversation summary
variables, and optionally a few labeled example images) into a vision-language
model — **Qwen2.5-VL** (7B or 32B) served via the SGLang offline engine — asking
for the clinical narrative or the item-level answer. It consumes the **same test
split** as LENS. `query_vlm.py` renders charts on the fly from each example's
`timeseries` field (or reads pre-built `*_test_idx{idx}_uid_{uid}.png` plots via
`--plots-dir`); `prompt.py` holds the templates; `eval_vlm_metrics.py` scores
the JSON/JSONL/CSV outputs with the standard NLP suite; `extract_ema_responses.py`
links each chart to its EMA item scores for the EMA-grounded judge. Small index
files (`shuffled_test_indices.json`, `ema_responses_mapping.json`) are kept here —
the latter is a **synthetic stub** (real EMA data is IRB-restricted; regenerate
with `extract_ema_responses.py`). Shell wrappers in `scripts/` (`start_sglang_vlm.sh`,
`run_vlm_query_in_container.sh`) take env-var paths and require GPUs. (Upstream
the code lived in a folder mis-named `Qwen3-VL`; the model is Qwen2.5-VL —
renamed to `ts_image` here.)
