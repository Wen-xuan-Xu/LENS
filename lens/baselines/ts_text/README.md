# `ts_text` — text-serialized few-shot baseline

The **TS-Text** baseline feeds the seven raw passive-sensing streams (heart rate,
pseudoactigraphy, steps, stress, GPS longitude, GPS latitude, phone unlock) plus
the sleep-duration and conversation-length summary variables into a generic LLM
*as plain text inside the prompt* (no time-series encoder), and asks for the
clinical narrative (`query_text_baseline.py`) or the item-level answer
(`query_qa_baseline.py`). It consumes the **same test split** as LENS. In the
paper the model is Qwen2.5-14B served via SGLang behind an OpenAI-compatible
endpoint — start it with `start_sglang.sh` (env-var paths: `MODEL_PATH`, `PORT`,
`TP_SIZE`, `SIF_IMAGE`, …). Outputs are JSONL with `{index, uid, prompt,
prediction, reference}`; score them with `lens.eval.metrics.compute_nlp_metrics`
and the `lens.eval.llm_judge` pipeline. No API key is required for a local
server; for a hosted endpoint pass `--api-key-env OPENAI_API_KEY` (the key is
read from the environment, never stored here).
