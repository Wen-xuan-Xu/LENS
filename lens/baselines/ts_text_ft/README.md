# `ts_text_ft` — text-only fine-tuned baseline (Qwen2.5-14B via LLaMA-Factory)

The **TS-Text-FT** ablation row: a text-only model (no time-series encoder)
fine-tuned on the LENS narrative + QA datasets, with the seven raw series
**stripped of their `<ts>` placeholder tokens and fed in as plain text**. It uses
the same test split as LENS, but `cutoff_len: 20000` (vs ~4000 for the encoder
runs) because the series-as-text prompts are long; `finetuning_type: lora`
(LoRA, the paper's configuration; set to `full` for a full SFT — see the YAML).

This is just **stock [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
plus a dataset-path pointer**. We vendor only:
- `qwen2_5_14b_text_ft.yaml` — the training config (placeholders for
  `${MODEL_DIR}` / `${DATA_ROOT}` / `${OUTPUT_DIR}`; `cutoff_len: 20000`,
  `finetuning_type: lora`, `template: qwen` — `<ts>` tokens stripped, series fed
  as text);
- `run_train.sh` — env-var-driven launcher (`MODEL_DIR`, `DATA_ROOT`,
  `OUTPUT_DIR`; runs `llamafactory-cli train`);
- `scripts/{check_overlength_samples,infer_merge_lora,sanity_infer_hf}.py` —
  small LENS helpers (overlength audit; LoRA-merge inference dump;
  HF sanity-check generation).

## 1. Install LLaMA-Factory

Use the latest [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory):

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
# (or: pip install "llamafactory[torch,metrics] @ git+https://github.com/hiyouga/LLaMA-Factory.git")
```

If you need byte-for-byte reproducibility later, pin whatever commit you used
(`git -C LLaMA-Factory rev-parse HEAD`).

## 2. Dataset registration (same jsonl as LENS training)

The text-FT baseline trains on the **same `narrative_dataset` / `qa_dataset`
jsonl produced by `make fake-data`** (or by the LENS pipeline on the private
study data) — see `lens/training/README.md`. Register them in your
LLaMA-Factory `data/dataset_info.json` with the **plain-text Alpaca columns**
(`prompt→instruction`, `query→input`, `response→output`) — no `timeseries`
column, because this baseline strips the `<ts>` tokens and the series live in
the `input` text:

```jsonc
"narrative_dataset": { "file_name": "narrative_dataset",
  "columns": { "prompt": "instruction", "query": "input", "response": "output" } },
"qa_dataset":        { "file_name": "qa_dataset",
  "columns": { "prompt": "instruction", "query": "input", "response": "output" } }
```

(If you also want the upstream `ift` / `align_random` channels in the mix, add
them the same way — but as the HF-Hub `ChatTSRepo/ChatTS-Training-Dataset`
subsets, exactly like `lens/training/dataset_info.template.json`.)

(Contrast with the LENS encoder runs, which register the same LENS datasets but
with a `timeseries` column and use the `chatts` template — a ChatML-style
template with no default system prompt — provided by the ChatTS-Training overlay.
The text-FT baseline just uses the stock `qwen` template.)

## 3. Run

```bash
export MODEL_DIR=Qwen/Qwen2.5-14B   # or a local path
export DATA_ROOT=/path/to/llamafactory/data   # dir holding dataset_info.json + the dataset folders
export OUTPUT_DIR=/path/to/checkpoints/ts_text_ft
bash run_train.sh

# inference -> NLP metrics -> LLM judge
accelerate launch -m lens.baselines.ts_text_ft.scripts.infer_merge_lora \
  --base-model "$MODEL_DIR" --adapter-path "$OUTPUT_DIR" --data-root "$DATA_ROOT" \
  --output-dir inference_results/ts_text_ft
python -m lens.eval.metrics.compute_nlp_metrics \
  --pred inference_results/ts_text_ft/narrative_results.jsonl \
  --pred-field prediction --ref-field reference --output metrics_ts_text_ft.json
```
