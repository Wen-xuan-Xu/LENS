#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coverage and Presence-aware Severity Alignment for narrative evaluations.

Consumes the merged structured-eval JSONL (one record per sample, with an
``evaluation`` dict mapping each of the 14 EMA symptom categories to
``ref_presence`` / ``pred_presence`` (0/1) and ``ref_severity`` /
``pred_severity`` (0..3) — produced by merging the ``--side ref`` and
``--side pred`` outputs of ``lens.eval.llm_judge.run_structured_eval``).

Reports:
  * Coverage: TP/FP/FN/TN + Precision/Recall/F1 over symptom *presence*.
  * Presence-aware alignment: 0.0 on any presence mismatch (hallucinated or
    missed symptom), otherwise ordinal-severity weight (0/1/2/3 -> 1/.75/.25/0);
    cases where both sides agree the symptom is absent are excluded.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_eval_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def flatten_entries(records: List[Dict]) -> List[Dict[str, int]]:
    entries: List[Dict[str, int]] = []
    for rec in records:
        for entry in rec.get("evaluation", {}).values():
            try:
                entries.append({
                    "ref_presence": int(entry["ref_presence"]),
                    "pred_presence": int(entry["pred_presence"]),
                    "ref_severity": int(entry["ref_severity"]),
                    "pred_severity": int(entry["pred_severity"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
    return entries


def compute_coverage(entries) -> Tuple[int, int, int, int, float, float, float]:
    tp = fp = fn = tn = 0
    for e in entries:
        rp, pp = e["ref_presence"], e["pred_presence"]
        if rp == 1 and pp == 1:
            tp += 1
        elif rp == 0 and pp == 1:
            fp += 1
        elif rp == 1 and pp == 0:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return tp, fp, fn, tn, prec, rec, f1


def compute_presence_aware_alignment(entries) -> Tuple[float, int]:
    weights = []
    for e in entries:
        rp, pp = e["ref_presence"], e["pred_presence"]
        rs, ps = e["ref_severity"], e["pred_severity"]
        if rp == 0 and pp == 0:
            continue
        if rp != pp:
            weights.append(0.0)
            continue
        diff = abs(rs - ps)
        weights.append(1.0 if diff == 0 else 0.75 if diff == 1 else 0.25 if diff == 2 else 0.0)
    return (sum(weights) / len(weights) if weights else 0.0), len(weights)


def evaluate_file(path: Path) -> Dict:
    records = list(load_eval_file(path))
    entries = flatten_entries(records)
    tp, fp, fn, tn, prec, rec, f1 = compute_coverage(entries)
    align_score, align_n = compute_presence_aware_alignment(entries)

    coverage: Dict = {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "Precision": prec, "Recall": rec, "F1": f1}
    alignment: Dict = {"PresenceAwareScore": align_score, "EvaluatedEntries": align_n}
    return {"file": str(path), "coverage": coverage, "alignment": alignment}


def main() -> None:
    p = argparse.ArgumentParser(description="Coverage + presence-aware severity alignment metrics.")
    p.add_argument("--files", nargs="+", required=True, help="Merged eval JSONL files.")
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-csv", type=Path, default=None)
    args = p.parse_args()

    results = []
    for f in args.files:
        r = evaluate_file(Path(f))
        results.append(r)
        print(f"\n=== {r['file']} ===")
        print(json.dumps(r, indent=2))
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] {args.out_json}")
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "PresenceAwareScore", "EvaluatedEntries"])
            for r in results:
                c, a = r["coverage"], r["alignment"]
                w.writerow([r["file"], c["TP"], c["FP"], c["FN"], c["TN"], c["Precision"], c["Recall"], c["F1"],
                            a["PresenceAwareScore"], a["EvaluatedEntries"]])
        print(f"[OK] {args.out_csv}")


if __name__ == "__main__":
    main()
