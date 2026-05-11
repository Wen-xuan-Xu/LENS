# LENS — training

Two-stage training of LENS (a patch-based time-series encoder + Qwen2.5-14B,
also a 7B variant), built on **ChatTS-Training** (a LLaMA-Factory fork with the
`chatts` template and a TS encoder), included as a git submodule under
`third_party/ChatTS-Training`. Needs GPUs and the submodule — **not run by CI.**

Model checkpoints (LENS-14B, LENS-7B) are released on the Hugging Face Hub; see
the top-level `README.md` for links.

## Layout

```
configs/training/stage1.yaml  configs/training/stage2.yaml  configs/training/lora.yaml
configs/training/deepspeed/   ZeRO-2 / ZeRO-3 (+ offload) configs the stages use
lens/training/scripts/        train_stage1.sh, train_stage2.sh
lens/training/lora/           train_lora.sh, ds_config_lora.json
lens/training/dataset_info.template.json   dataset registration to copy into the submodule
```

## 1. Initialize the submodule

```bash
git submodule update --init --recursive third_party/ChatTS-Training
# (then pin a commit — see third_party/README.md)
pip install -e "third_party/ChatTS-Training[.]"   # or follow that repo's install docs
```

## 2. Prepare data

LENS uses five datasets — `narrative_dataset`, `qa_dataset`, `align_256`,
`align_random`, `ift` — all with columns `{input, output, timeseries}` (the
`timeseries` column = the per-sample list of raw sensor windows in the fixed
order `[hr_window, zcr_prod, steps_first, stress_window, gps_lon, gps_lat,
unlock]`, lengths `[1440, 480, 240, 240, 24, 24, 240]`, one per `<ts><ts/>`
placeholder in `input`). Two come from LENS, three come from upstream:

* **`narrative_dataset` / `qa_dataset` — LENS-generated**, kept as local
  `file_name` paths. Synthetic / runnable: `make fake-data` then
  `make smoke-test-pipeline` produces these under `data/fake/`. Real study data:
  build the jsonl with the `lens.data_pipeline` tools pointed at your
  `--data-root` (study data is not distributed — see the top-level README).
* **`align_256` / `align_random` / `ift` — ChatTS upstream**, hosted on the HF
  Hub at [`ChatTSRepo/ChatTS-Training-Dataset`](https://huggingface.co/datasets/ChatTSRepo/ChatTS-Training-Dataset)
  (Apache-2.0; in that repo each subset is a subfolder containing a single
  `train.jsonl`). `dataset_info.template.json` registers them via LLaMA-Factory's
  `hf_hub_url` form (`hf_hub_url: ChatTSRepo/ChatTS-Training-Dataset`,
  `folder: <subset>`, `split: train`), so **LLaMA-Factory downloads them
  automatically** when training starts — nothing to copy by hand. You may need
  `huggingface-cli login` first. (The fake registry `data/fake/dataset_info.json`
  instead aliases `align_256` to a tiny local `align_random` so the smoke test
  runs offline.)

Register them inside the submodule:

```bash
cp lens/training/dataset_info.template.json third_party/ChatTS-Training/data/dataset_info.json
# (or merge its entries into the submodule's existing data/dataset_info.json)
# then symlink/copy your LENS jsonl dirs into third_party/ChatTS-Training/data/, e.g.
ln -s "$PWD/data/fake/narrative_dataset" third_party/ChatTS-Training/data/narrative_dataset
ln -s "$PWD/data/fake/qa_dataset"        third_party/ChatTS-Training/data/qa_dataset
# align_256 / align_random / ift need no local files — they pull from the HF Hub.
```

## 3. Set env vars and run

The launch scripts auto-detect the repo root and the submodule; override paths
via env vars (see the header of each script). Minimum you usually set:

```bash
export LENS_MODEL_DIR=Qwen/Qwen2.5-14B   # or a local path; Qwen/Qwen2.5-7B for the 7B variant
export LENS_NUM_GPUS=8
export LENS_OUTPUT_DIR="$PWD/models"

bash lens/training/scripts/train_stage1.sh        # encoder alignment  -> $LENS_OUTPUT_DIR/lens-stage1
LENS_MODEL_DIR="$LENS_OUTPUT_DIR/lens-stage1" \
  bash lens/training/scripts/train_stage2.sh      # supervised fine-tune -> $LENS_OUTPUT_DIR/lens-stage2
# (or: make train-stage1 && make train-stage2)
```

Other env vars: `LENS_CHATTS_DIR`, `LENS_DATA_DIR` (default `data`, relative to
the submodule), `LENS_DEEPSPEED`, `LENS_MASTER_PORT`, `LENS_STAGE_YAML`.
For memory-tight nodes point `LENS_DEEPSPEED` at the `*_offload.json` variants.

LoRA variant (frozen backbone, LoRA adapters + trainable TS encoder):

```bash
bash lens/training/lora/train_lora.sh             # -> $LENS_OUTPUT_DIR/lens-lora
```

## Stage ratios

| Stage | Datasets | Interleave probs | `cutoff_len` | Trains |
|-------|----------|------------------|--------------|--------|
| 1 — encoder alignment | `align_256 : narrative_dataset : qa_dataset` | `0.8 : 0.1 : 0.1` | 4000 | TS encoder + projector (backbone effectively frozen) |
| 2 — supervised fine-tuning | `narrative_dataset : qa_dataset : ift : align_random` | `0.3 : 0.3 : 0.2 : 0.2` | 4000 | full Qwen2.5-14B/7B backbone + encoder |
| LoRA variant | same as Stage 2 | `0.3 : 0.3 : 0.2 : 0.2` | 4000 | LoRA adapters + TS encoder |

(The TS-Text-FT text-only baseline uses `cutoff_len 20000` and lives in
`lens/baselines/ts_text_ft/`, not here.)
