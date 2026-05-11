# `lens/feature_engineering/`

Turns raw passive-sensing CSVs + EMA self-reports into the per-EMA feature
table the dataset-build stage consumes. The real LENS study data is not
distributable; everything here runs on the synthetic data under `data/fake/`.

## Pipeline

1. `build_feature_rows.py` — for each EMA, slice the preceding 4 h of every
   sensor stream (via `searchsorted` on sorted timestamps) → `feature_rows.csv`,
   one row per EMA.
   ```
   python -m lens.feature_engineering.build_feature_rows \
       --data-root data/fake --output data/fake/feature_rows.csv
   ```
2. `add_unlock_and_narratives.py` — adds the per-minute phone-unlock column
   (1 Hz → "any unlock per 60 s"), recomputes the rule-based summary narrative
   from the EMA items (`ema_codebook.summary_narrative`), and left-joins the
   LLM-rewritten `enhanced_narrative` (if `enriched_narratives.jsonl` exists) →
   `filtered_feature_rows.csv`.
   ```
   python -m lens.feature_engineering.add_unlock_and_narratives \
       --data-root data/fake --in data/fake/feature_rows.csv \
       --out data/fake/filtered_feature_rows.csv
   ```

`--data-root` resolves either layout: raw channels under `<root>/raw_sensors/`
(public) or `<root>/raw_data/` (private); sleep under
`<root>/garmin_health_api/sleep/`.

Helpers: `extract_enhanced_fields.py` (slim the enriched JSONL to just the
rewritten text), `data_integrity_check.py` (assert per-channel lengths),
`ema_codebook.py` (EMA item inventory, 0–100 → quartile → label maps, PHQ-9
scoring, rule-based templates), `legacy/ema_sensor_matcher.py` (older, superseded
matcher — reference only).

## `feature_rows.csv` columns

`uid, response_time, ema_Q1..ema_Q14, ema_Q11(0), ema_Q11(1), ema_Q11(2),
hr_window, hr_count, zcr_first, zcr_last, zcr_count, steps_first, steps_count,
stress_window, stress_count, gps_lat, gps_lon, gps_count, convo_duration,
sleep_duration`. `add_unlock_and_narratives` appends `unlock_min`,
`unlock_min_count`, `rule_based_narrative`, `enhanced_narrative`.

## Sensor sampling rates (paper Appendix C; cross-check CLAUDE.md §3.1)

| Stream | Rate | Length over 4 h |
|---|---|---|
| Heart rate (bpm) | 10 s | 1440 |
| Pseudoactigraphy (ZCR count × energy) | 30 s | 480 |
| Steps / minute | 60 s | 240 |
| Stress level (Garmin HRV-derived) | 60 s | 240 |
| GPS longitude | 10 min | 24 |
| GPS latitude | 10 min | 24 |
| Phone unlock (0/1 per minute) | 60 s | 240 |
| Sleep duration (h) | scalar | 1 (EMA-day max) |
| Conversation length (s) | scalar | 1 (summed over window) |
