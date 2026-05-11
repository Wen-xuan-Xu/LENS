# `lens.eval` — inference + metrics + LLM-as-judge

Evaluation for LENS narratives (summary-level) and QA answers (item-level), on
the **same test split** as training. Three layers:

- `lens/eval/inference/` — generate predictions from a trained checkpoint:
  - `hf_inference.py` — HF Transformers, multi-GPU via `accelerate`, for the
    patch-based TS-encoder models. Writes `<dataset>_results.jsonl` with
    `{dataset, index, reference, prediction}`. (GPUs required.)
- `lens/eval/metrics/` — reference-based NLP metrics:
  - `compute_nlp_metrics.py` — **ROUGE-1/2/L (P/R/F1), BLEU-1..4, METEOR,
    BERTScore (P/R/F1)** (point estimates). Heavy deps (`rouge-score`,
    `sacrebleu`, `nltk`, `bert-score`) are lazy-imported and **skipped
    gracefully** if absent. Has a `--self-test` flag (used by CI):
    `python -m lens.eval.metrics.compute_nlp_metrics --pred examples/one_sample_each.jsonl --ref examples/one_sample_each.jsonl --self-test`
  - `compute_qa_metrics.py` — **Severity Alignment** for QA (`|ref-pred|` → 1/.75/.25/0).
  - `compute_custom_metrics.py` — **Coverage** (Precision/Recall/F1 over symptom
    presence) + **Presence-aware Severity Alignment** for narratives.
  - `eval_bleu_rouge.py` — thin LLaMA-Factory-style BLEU-4/ROUGE over a
    `generated_predictions.jsonl` (needs `jieba`, `rouge-chinese`).
  - `visualize_metrics.py` — render Markdown comparison tables from metrics JSONs.
- `lens/eval/llm_judge/` — LLM-as-a-Judge (model: **`gpt-4.1-mini`**) over the
  OpenAI Batch API. Three axes: **Coverage**, **Presence Alignment**, **Severity
  Alignment**.
  - `run_structured_eval.py` — per-text, rate `{presence, severity}` for the 14
    EMA symptom categories. Run `--side ref` and `--side pred`, then `--merge`.
  - `run_qa_eval.py` — per QA pair, emit `{ref_severity, pred_severity}` (0..3).
- `lens/eval/scripts/` — shell wrappers (env-var paths; header comments note
  GPU/API needs): `run_llm_judge_pipeline.sh`, `run_qa_eval_batch.sh`,
  `eval_half_data.sh`, `run_ablation_hf_inference_suite.sh`,
  `run_stage2_ablation_suite.sh`.

## Pipeline

```
checkpoint --(hf_inference)--> <dataset>_results.jsonl
  ├─ compute_nlp_metrics.py            -> ROUGE / BLEU / METEOR / BERTScore
  └─ run_structured_eval (ref & pred) --merge--> compute_custom_metrics.py   (narrative)
     run_qa_eval.py                  --------->  compute_qa_metrics.py        (QA)
```

## Environment

- `pip install -e .[eval]` for the metric/judge extras (`openai`, `pydantic`,
  `rouge-score`, `nltk`, `bert-score`, `sacrebleu`). Core (`numpy`) is enough for
  `--self-test`.
- `OPENAI_API_KEY` — required for the LLM-as-judge scripts. **No key is stored in
  this repo**; everything reads it from the environment.
- `DATA_ROOT`, `MODEL_DIR` — point inference scripts at your data / checkpoint
  (default `DATA_ROOT=data/fake`). NLTK may need `punkt` / `wordnet` for METEOR
  (`python -m nltk.downloader punkt wordnet omw-1.4`).
