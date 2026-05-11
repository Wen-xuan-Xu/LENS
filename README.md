# LENS: LLM-Enabled Narrative Synthesis for Mental Health

Aligning multimodal passive sensing with language models to generate clinically
grounded mental-health narratives.

> Xu, Pillai, Nepal, Collins, Mackin, Heinz, Griffin, Jacobson, Campbell.
> *LENS: LLM-Enabled Narrative Synthesis for Mental Health by Aligning
> Multimodal Sensing with Language Models.*

LENS turns raw wearable/phone sensor streams (heart rate, accelerometer-derived
ZCR, steps, stress, GPS, phone unlock, sleep, conversation) plus Ecological
Momentary Assessment (EMA) self-reports into natural-language summaries of a
person's recent depression/anxiety symptom state. It has two parts:

1. **A data-synthesis pipeline** — converts EMA responses into item-level and
   summary-level template narratives, refines them with an LLM, and filters them
   with a multi-agent LLM-as-a-judge. Produces ~102k item-level QA pairs and
   ~51k summary-level narratives, each paired with the corresponding raw
   sensor windows.
2. **A patch-based time-series encoder + LLM** — a lightweight MLP encoder maps
   each sensor stream into the LLM's embedding space; the model is trained in
   two stages (encoder alignment, then supervised fine-tuning on the
   EMA-derived datasets), built on
   [ChatTS-Training](https://github.com/xiezhe-24/ChatTS-Training).

## ⚠️ Data availability

The passive-sensing and EMA data underlying the paper were collected under an
NIH/NIMH award (R01MH123482-01) and are governed by IRB and funder data-use
restrictions — **they are not distributed in this repository.** De-identified
study data is the responsibility of the [NIMH NDA archive](https://nda.nih.gov).

To keep the code runnable, this repo ships **only a deterministic synthetic-data
generator** — `data/fake/generate.py` (pinned seed). Run `make fake-data` to
materialize a complete, schema-faithful fake dataset under `data/fake/` and
`examples/` (raw sensor CSVs at the paper's sampling rates, synthetic EMA, the
rule→rewrite→judge intermediates, and the training-ready jsonl). None of that
generated data is committed — regenerate it any time. Every script also accepts
a `--data-root` so the same code runs against the private study data. See
`data/fake/README.md`.

## Repository layout

```
lens/feature_engineering/  raw sensors + EMA  ->  windowed feature rows (feature_rows.csv)
lens/data_pipeline/        EMA -> rule templates -> LLM enrichment -> multi-judge QC -> training jsonl
lens/training/             two-stage SFT of LENS-14B / LENS-7B  (uses third_party/ChatTS-Training)
lens/eval/                 inference + ROUGE/BLEU/METEOR/BERTScore + LLM-as-judge (coverage / presence / severity)
lens/baselines/            ts_text (few-shot), ts_image (Qwen2.5-VL), ts_text_ft (text-only fine-tune via LLaMA-Factory)
configs/                   YAML configs (no secrets; API keys read from env)
data/fake/generate.py      deterministic synthetic-data generator (run `make fake-data`)
third_party/ChatTS-Training git submodule (the training stack)
```

## Quickstart

```bash
git clone --recursive https://github.com/Wen-xuan-Xu/LENS.git
cd LENS
pip install -e ".[pipeline,eval,dev]"

# materialize the synthetic data, then run the whole thing end-to-end
make fake-data
make smoke-test
```

Pipeline stages individually:

```bash
make smoke-test-feature    # raw sensors -> feature_rows.csv -> filtered_feature_rows.csv
make smoke-test-pipeline   # EMA -> templates -> (mock) LLM enrichment -> dataset_build -> jsonl
make smoke-test-eval       # (mock) inference -> NLP metrics
```

### Sensor sampling rates (used by the synthetic data, matching the paper)

After preprocessing, all streams are standardized over a 4-hour pre-EMA window:
heart rate every 10 s (length 1440), pseudoactigraphy = ZCR count × energy
every 30 s (480), steps every 60 s (240), stress every 60 s (240), GPS
longitude/latitude every 10 min (24 each), phone unlock every 60 s (240); sleep
duration (h) and conversation length (s) are stored as scalars per window.

## Training

Training uses the [ChatTS-Training](https://github.com/xiezhe-24/ChatTS-Training)
stack (a fork of LLaMA-Factory with the `chatts` template and a patch-based
time-series encoder), pinned here as a git submodule under
`third_party/ChatTS-Training`. The LENS-specific launch scripts, YAML configs,
and DeepSpeed configs live in `lens/training/`. See `lens/training/README.md`.

- **Stage 1 (encoder alignment):** `align_256 : narrative : qa = 0.8 : 0.1 : 0.1`,
  `cutoff_len 4000`.
- **Stage 2 (supervised fine-tuning):** `narrative : qa : ift : align_random = 0.3 : 0.3 : 0.2 : 0.2`,
  `cutoff_len 4000`, full-parameter fine-tuning of the Qwen2.5-14B backbone +
  encoder. (`cutoff_len 20000` for the TS-Text-FT text-only baseline.)

Model checkpoints (LENS-14B, LENS-7B) will be released on the Hugging Face Hub;
links will be added here.

## Citation

```bibtex
@inproceedings{xu2025lens,
  title     = {LENS: LLM-Enabled Narrative Synthesis for Mental Health by Aligning Multimodal Sensing with Language Models},
  author    = {Xu, Wenxuan and Pillai, Arvind and Nepal, Subigya and Collins, Amanda C. and Mackin, Daniel M. and Heinz, Michael V. and Griffin, Tess Z. and Jacobson, Nicholas C. and Campbell, Andrew},
  year      = {2025}
}
```

If you use the training stack or the alignment data, please also cite ChatTS
(Xie et al., 2025) and LLaMA-Factory.

## License

Apache-2.0 (see `LICENSE` and `NOTICE`).
