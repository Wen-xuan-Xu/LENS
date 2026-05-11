#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Severity-Alignment score for item-level QA evaluations.

Reads the JSONL produced by ``lens.eval.llm_judge.run_qa_eval`` (records with
``ref_severity`` / ``pred_severity`` integers in 0..3) and computes the
Severity Alignment Score described in the paper:

    diff = |ref_severity - pred_severity|
    diff == 0 -> 1.00 ; diff == 1 -> 0.75 ; diff == 2 -> 0.25 ; diff >= 3 -> 0.00
    score = mean(weights)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_eval_file(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _weight(diff: int) -> float:
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.75
    if diff == 2:
        return 0.25
    return 0.0


def compute_severity_alignment(records: List[dict]) -> Tuple[float, int, Dict[str, int]]:
    weights: List[float] = []
    breakdown = {"diff_0": 0, "diff_1": 0, "diff_2": 0, "diff_3plus": 0}
    for rec in records:
        rs, ps = rec.get("ref_severity"), rec.get("pred_severity")
        if rs is None or ps is None:
            continue
        diff = abs(int(rs) - int(ps))
        weights.append(_weight(diff))
        if diff == 0:
            breakdown["diff_0"] += 1
        elif diff == 1:
            breakdown["diff_1"] += 1
        elif diff == 2:
            breakdown["diff_2"] += 1
        else:
            breakdown["diff_3plus"] += 1
    score = sum(weights) / len(weights) if weights else 0.0
    return score, len(weights), breakdown


def evaluate_file(path: Path) -> Dict:
    records = load_eval_file(path)
    score, evaluated, breakdown = compute_severity_alignment(records)
    out: Dict = {
        "file": str(path),
        "total_records": len(records),
        "evaluated_entries": evaluated,
        "severity_alignment_score": score,
        "breakdown": breakdown,
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Compute QA Severity-Alignment metrics.")
    p.add_argument("--files", nargs="+", required=True, help="QA eval JSONL files to score.")
    p.add_argument("--output", type=Path, default=None, help="Optional JSON output.")
    args = p.parse_args()

    results = []
    for f in args.files:
        r = evaluate_file(Path(f))
        results.append(r)
        print(f"\n=== {r['file']} ===")
        print(json.dumps(r, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[OK] metrics saved to {args.output}")


if __name__ == "__main__":
    main()
