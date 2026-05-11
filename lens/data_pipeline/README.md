# `lens/data_pipeline/` — data-synthesis pipeline

Turns EMA self-reports into the clinically grounded training narratives LENS is
trained on (paper Section 3.1; cross-check `CLAUDE.md` §3.2).

## Stages

1. **Rule-based templates** (`template_enrichment/generator.py`,
   `EMATemplateGenerator`) — each EMA (the 14-item PHQ-9 + GAD-derived
   self-report) becomes a deterministic *summary-level* narrative and a set of
   *item-level* answer sentences.
2. **LLM rewriting** (`template_enrichment/{prompt,sglang_batch_processor}.py`)
   — an instruction-tuned LLM (Qwen2.5-14B, served via SGLang's
   OpenAI-compatible API) rewrites each template into a fluent narrative while
   preserving every fact and severity level. Driven by `run_pipeline.py`
   (`--mock-llm` substitutes a deterministic stub so the pipeline runs offline).
3. **Multi-agent LLM-as-a-judge QC** (`llm_evaluation/`) — an ensemble of three
   small open models (Mistral-7B, Llama-3.1-8B, Qwen2.5-7B) scores each rewrite
   against its template on five 1–5 dimensions (Factual Alignment, Symptom
   Coverage, Severity Fidelity, Fluency & Naturalness, Hallucination Risk);
   confidence-weighted voting → PASS/FAIL; PASS narratives become labels.
   `parallel_judge.py` deploys all judges at once; `sequential_judge.py` does
   one at a time (lower VRAM).
4. **Dataset build chain** (`dataset_build/`, `fix_ts_tokens.py`):
   `filtered_feature_rows.csv` + `Questions.json` + enriched JSONLs
   → `build_dataset.py` → HF Arrow `DatasetDict` (`{uid, input, timeseries,
   output}`, splits = train/validation/test, 70/15/15 by participant,
   placeholder `<ts></ts>`)
   → `convert_hf_to_jsonl.py` → JSONL splits (still `<ts></ts>`)
   → `fix_ts_tokens.py` → final JSONL with `<ts><ts/>` (what stage-1/2 SFT
   consumes).

## EMA discretization (paper Appendix B)

Items scored 0–100, mapped to four equal bands: `0–25 → "not at all"`,
`26–50 → "sometimes"`, `51–75 → "often"`, `76–100 → "constantly"`. Q3 (sleep,
"last night") uses an intensity scale `minimal / moderate / significant /
severe`. The PHQ-9 total folds each Q1–Q9 value onto the standard 0–3 quartiles,
summed, then interpreted with the usual severity bands.

## Time-series streams

Each sample carries 7 streams (= the 7 `<ts><ts/>` placeholders), in order with
lengths over the 4-hour pre-EMA window: `hr_window` (1440) · `zcr_prod` =
ZCR-count × energy (480) · `steps_first` (240) · `stress_window` (240) ·
`gps_lon` (24) · `gps_lat` (24) · `unlock` (240).

## LENS-generated vs ChatTS-upstream

`narrative_dataset` and `qa_dataset` are produced by this pipeline. The
`align_256` / `align_random` / `ift` training channels come from the **ChatTS
upstream** release, not here (`data/fake/{align_random,ift}/` ship stand-ins).

## Quickstart (offline, on fake data)

```bash
python data/fake/generate.py --out data/fake --all
python -m lens.data_pipeline.run_pipeline --config configs/pipeline/smoke.yaml --mock-llm
python -m lens.data_pipeline.dataset_build.build_dataset --config configs/pipeline/dataset_build_smoke.yaml
python -m lens.data_pipeline.dataset_build.convert_hf_to_jsonl --root data/fake/arrow --out data/fake
python -m lens.data_pipeline.fix_ts_tokens data/fake/narrative_dataset data/fake/qa_dataset
```

The LLM-backed stages need the `pipeline` extra (`openai`, `pydantic`,
`sglang`); `--mock-llm` and the build chain run with only the core deps.
No API keys live in this repo — see `configs/pipeline/config.example.yaml`.
