#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the EMA-responses mapping that links each TS-Image chart to its EMA item scores.

For every rendered test chart (``*_test_idx{idx}_uid_{uid}.png``), this looks up
the corresponding dataset sample by ``(uid, output[:100])`` in the enriched-JSONL
files to recover ``ema_timestamp``, then joins on ``(uid, ema_timestamp ==
response_time)`` in the feature-rows CSV to pull the 14 EMA item scores. Writes a
``ema_responses_mapping.json`` (and per-dataset CSVs) used downstream to compute
EMA-grounded judge scores.

All paths are CLI/env-var driven (no hardcoded ``/scratch`` paths). On the public
release the real EMA data is not distributed; ``ema_responses_mapping.json`` here
is a tiny **synthetic** stub showing the schema — re-run this script against your
own data to regenerate it.

Example
-------
    python -m lens.baselines.ts_image.extract_ema_responses \
        --narrative-jsonl "$DATA_ROOT/narrative_dataset/test.jsonl" \
        --qa-jsonl "$DATA_ROOT/qa_dataset/test.jsonl" \
        --feature-csv "$DATA_ROOT/filtered_feature_rows.csv" \
        --enriched-narratives "$DATA_ROOT/enriched_narratives.jsonl" \
        --enriched-qas "$DATA_ROOT/enriched_qas.jsonl" \
        --narrative-plots-dir out/narrative_test_plots \
        --qa-plots-dir out/qa_test_plots \
        --output out/ema_responses_mapping.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

EMA_COLUMNS = [
    "ema_Q1", "ema_Q2", "ema_Q3", "ema_Q4", "ema_Q5", "ema_Q6", "ema_Q7", "ema_Q8",
    "ema_Q9", "ema_Q10", "ema_Q11(0)", "ema_Q11(1)", "ema_Q11(2)", "ema_Q12", "ema_Q13", "ema_Q14",
]
_PLOT_RE = re.compile(r"_test_idx(?P<idx>\d+)_uid_(?P<uid>.+)\.png$")


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_enriched(path: Path, text_key: str) -> Dict:
    out = {}
    for obj in _read_jsonl(path):
        out[(obj.get("uid"), (obj.get(text_key, "") or "")[:100])] = obj
    return out


def _scan_plots(plots_dir: Optional[Path]) -> List[tuple]:
    if not plots_dir or not plots_dir.exists():
        return []
    out = []
    for f in sorted(plots_dir.glob("*.png")):
        m = _PLOT_RE.search(f.name)
        if m:
            out.append((f.name, int(m.group("idx")), m.group("uid")))
    return out


def _extract_for(sample: dict, enriched: Dict, dataset_type: str, feature_index) -> Optional[dict]:
    uid, output = sample.get("uid"), sample.get("output", "")
    entry = enriched.get((uid, output[:100]))
    if not entry:
        text_key = "enhanced_answer" if dataset_type == "qa" else "enhanced_narrative"
        for (suid, _), e in enriched.items():
            if suid == uid and e.get(text_key, "") == output:
                entry = e
                break
    if not entry:
        return None
    ema_timestamp = entry.get("ema_timestamp")
    if not ema_timestamp:
        return None
    row = feature_index.get((uid, ema_timestamp))
    if row is None:
        return None
    ema_responses = {c: row.get(c) for c in EMA_COLUMNS if c in row}
    extra = ({"rule_based_answer": entry.get("rule_based_answer", ""), "enhanced_answer": entry.get("enhanced_answer", "")}
             if dataset_type == "qa" else
             {"rule_based_narrative": entry.get("rule_based_narrative", ""), "enhanced_narrative": entry.get("enhanced_narrative", "")})
    return {"uid": uid, "ema_timestamp": ema_timestamp, "dataset_type": dataset_type,
            "output": output, "ema_responses": ema_responses, **extra}


def main() -> None:
    p = argparse.ArgumentParser(description="Link TS-Image charts to EMA item scores.")
    dr = Path(os.environ.get("DATA_ROOT", "data/fake"))
    p.add_argument("--narrative-jsonl", type=Path, default=dr / "narrative_dataset" / "test.jsonl")
    p.add_argument("--qa-jsonl", type=Path, default=dr / "qa_dataset" / "test.jsonl")
    p.add_argument("--feature-csv", type=Path, default=dr / "filtered_feature_rows.csv")
    p.add_argument("--enriched-narratives", type=Path, default=dr / "enriched_narratives.jsonl")
    p.add_argument("--enriched-qas", type=Path, default=dr / "enriched_qas.jsonl")
    p.add_argument("--narrative-plots-dir", type=Path, default=None)
    p.add_argument("--qa-plots-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    import pandas as pd  # heavy; imported here so --help is light

    feature_df = pd.read_csv(args.feature_csv)
    feature_index = {}
    rt_col = "response_time" if "response_time" in feature_df.columns else None
    if rt_col:
        for _, r in feature_df.iterrows():
            feature_index[(r.get("uid"), r.get(rt_col))] = r.to_dict()

    enriched_narr = _load_enriched(args.enriched_narratives, "enhanced_narrative") if args.enriched_narratives.exists() else {}
    enriched_qa = _load_enriched(args.enriched_qas, "enhanced_answer") if args.enriched_qas.exists() else {}

    narrative_samples = _read_jsonl(args.narrative_jsonl) if args.narrative_jsonl.exists() else []
    qa_samples = _read_jsonl(args.qa_jsonl) if args.qa_jsonl.exists() else []

    results = {"narrative": [], "qa": []}
    for kind, samples, enriched, plots_dir in [
        ("narrative", narrative_samples, enriched_narr, args.narrative_plots_dir),
        ("qa", qa_samples, enriched_qa, args.qa_plots_dir),
    ]:
        plot_list = _scan_plots(plots_dir) or [(None, i, s.get("uid")) for i, s in enumerate(samples)]
        for plot_file, idx, _uid in plot_list:
            if idx >= len(samples):
                continue
            rec = _extract_for(samples[idx], enriched, kind, feature_index)
            if rec:
                rec["plot_file"] = plot_file
                rec["test_index"] = idx
                results[kind].append(rec)

    results["metadata"] = {
        "successfully_extracted_narrative": len(results["narrative"]),
        "successfully_extracted_qa": len(results["qa"]),
        "ema_columns": EMA_COLUMNS,
        "extraction_date": datetime.now().isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {args.output}: narrative={len(results['narrative'])}, qa={len(results['qa'])}")


if __name__ == "__main__":
    main()
