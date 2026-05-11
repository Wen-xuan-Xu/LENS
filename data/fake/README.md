# `data/fake/` — synthetic data

The real LENS study data (passive sensing + EMA from an NIH/NIMH-funded study)
is governed by IRB / funder data-use restrictions and is **not** distributed
here (see the repo `README.md`). To keep every script runnable, this directory
holds a single deterministic generator:

```
data/fake/generate.py
```

Run it (or `make fake-data`) to materialize a complete, schema-faithful
synthetic dataset under `data/fake/` and `examples/`:

```bash
make fake-data
# == python data/fake/generate.py --out data/fake --all
```

That produces (all .gitignored — regenerate any time):

- `raw_sensors/{hr,zcr,steps,stress,gps,convo,unlock}/...` and
  `garmin_health_api/sleep/...` — raw sensor CSVs at the paper's sampling rates
  (HR 10 s, ZCR 30 s, steps 60 s, stress 60 s, GPS 10 min, unlock 1 s, sleep/conversation scalars).
- `ema.csv`, `filtered_uids.json` — synthetic EMA responses (4 obviously-fake
  participants `fake_0001@demo` … `fake_0004@demo`, biased toward low/minimal symptom levels).
- `Questions.json` — the EMA item / summary question bank.
- `enriched_narratives.jsonl`, `enriched_qas.jsonl` — stand-ins for the LLM-rewrite outputs.
- `narrative_dataset/`, `qa_dataset/`, `align_random/`, `ift/` — training-ready jsonl
  (`<ts><ts/>` placeholders, 7 timeseries streams, participant-level 70/15/15 split).
- `feature_rows.csv`, `filtered_feature_rows.csv`, `arrow/` — pipeline intermediates.
- `dataset_info.json`, `SEED` — dataset registry + the pinned seed used.
- `../examples/one_sample_each.jsonl` — one truncated sample of each kind, for inspection.

The seed is pinned in `generate.py` (`SEED_DEFAULT`), so two runs are byte-identical.
Pass `--seed N` / `--out DIR` to vary it. The fake data is intentionally bland
(no clinical extremes); every script also accepts a `--data-root` so the same
code runs against the private study data.
