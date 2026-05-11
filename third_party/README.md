# third_party/

LENS does not vendor its training stack. The two-stage trainer (and the
patch-based time-series encoder, the `chatts` chat template, and the
`timeseries` dataset column) live in **ChatTS-Training**, a fork of
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) maintained by the
ChatTS authors. It is included here as a **git submodule**:

```
third_party/ChatTS-Training   ->  https://github.com/xiezhe-24/ChatTS-Training
```

Upstream lineage: ChatTS-Training (a LLaMA-Factory fork) ← ChatTS
(<https://github.com/NetmanAIOps/ChatTS>).

## Why a submodule (not vendored)

* It is a large third-party tree under its own license; copying it wholesale
  would bloat this repo and make upstream updates painful.
* The LENS-specific pieces (`chatts` template, `narrative_dataset`/`qa_dataset`/
  `align_*`/`ift` registrations with the `timeseries` column, the TS encoder)
  are already merged into the fork, so there is normally **nothing to patch** —
  you just register the LENS jsonl datasets and point the launch scripts at it.
* Model weights are *not* committed anywhere in this repo; LENS-14B / LENS-7B
  checkpoints are released on the Hugging Face Hub (links in the top-level
  `README.md`).

## Setup

This repo ships a hand-written `.gitmodules` (the maintainer ran `git init`
after assembly). Initialize the submodule and pin it:

```bash
git submodule update --init --recursive third_party/ChatTS-Training
# or, if adding from scratch:
#   git submodule add https://github.com/xiezhe-24/ChatTS-Training third_party/ChatTS-Training

# pin to a known-good commit (record the SHA you tested against here):
cd third_party/ChatTS-Training
git checkout <PINNED_COMMIT_SHA>   # TODO: maintainer fills this in after a test run
```

## What LENS overrides on top of the submodule

* Launch scripts / YAML / DeepSpeed configs: `lens/training/` and
  `configs/training/` (env-var-parameterized; never reference absolute paths).
* Dataset registration: copy `lens/training/dataset_info.template.json` into
  `third_party/ChatTS-Training/data/dataset_info.json` (or merge its entries),
  then symlink/copy the LENS jsonl datasets into
  `third_party/ChatTS-Training/data/`. See `lens/training/README.md`.

Do **not** commit anything generated inside the submodule (checkpoints,
`models/output-*/`, `env/`, `tmp/`, `*.safetensors`, `*.bin`).
