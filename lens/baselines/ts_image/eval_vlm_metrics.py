#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NLP metrics for TS-Image (VLM) outputs — JSON / JSONL / CSV.

Consolidates the three near-duplicate scripts from the upstream ``Qwen3-VL``
folder (``eval_metrics_VLM.py``, ``eva_metrics.py``, ``eva_metrics_jsonl.py``)
into one loader that feeds ``lens.eval.metrics.compute_nlp_metrics`` (ROUGE /
BLEU / METEOR / BERTScore). Heavy NLP deps are skipped gracefully if not
installed.

Accepted inputs:
  * JSONL: each line has prediction (``prediction`` | ``vlm_response`` | ``text``)
    and reference (``reference`` | ``ground_truth`` | ``output``).
  * JSON: a dict with a ``results`` list (each item like above), or a single
    ``{prediction|vlm_response, reference|ground_truth}`` dict.
  * CSV: columns named via ``--prediction-column`` / ``--reference-column``.

Example
-------
    python -m lens.baselines.ts_image.eval_vlm_metrics \
        --input out/vlm_responses_32b.jsonl --output out/vlm_metrics.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

from lens.eval.metrics.compute_nlp_metrics import evaluate, preprocess_text, print_summary

_PRED_KEYS = ("prediction", "vlm_response", "predict", "text")
_REF_KEYS = ("reference", "ground_truth", "output", "label")


def _pick(d: dict, keys: Tuple[str, ...]):
    for k in keys:
        if d.get(k) not in (None, ""):
            return d[k]
    return None


def load_pairs(path: Path, prediction_column: str | None, reference_column: str | None) -> Tuple[List[str], List[str]]:
    preds: List[str] = []
    refs: List[str] = []
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                p = row.get(prediction_column) if prediction_column else _pick(row, _PRED_KEYS)
                r = row.get(reference_column) if reference_column else _pick(row, _REF_KEYS)
                if p in (None, "") or r in (None, ""):
                    continue
                preds.append(preprocess_text(p))
                refs.append(preprocess_text(r))
    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                p, r = _pick(rec, _PRED_KEYS), _pick(rec, _REF_KEYS)
                if p is None or r is None:
                    raise ValueError(f"{path}:{i} missing prediction/reference fields")
                preds.append(preprocess_text(p))
                refs.append(preprocess_text(r))
    else:  # .json
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("results") if isinstance(data, dict) else None
        if items is None:
            items = [data] if isinstance(data, dict) and ("prediction" in data or "vlm_response" in data) else []
        if not items:
            raise ValueError(f"{path}: no `results` list or prediction/reference dict")
        for idx, it in enumerate(items):
            if not isinstance(it, dict):
                continue
            p, r = _pick(it, _PRED_KEYS), _pick(it, _REF_KEYS)
            if p is None or r is None:
                raise ValueError(f"{path} results[{idx}] missing prediction/reference")
            preds.append(preprocess_text(p))
            refs.append(preprocess_text(r))
    if not preds:
        raise ValueError(f"{path}: no (prediction, reference) pairs found")
    return preds, refs


def main() -> None:
    p = argparse.ArgumentParser(description="NLP metrics for TS-Image (VLM) outputs.")
    p.add_argument("--input", type=Path, required=True, help="JSON / JSONL / CSV with prediction & reference.")
    p.add_argument("--output", type=Path, default=None, help="Optional metrics JSON output path.")
    p.add_argument("--prediction-column", type=str, default=None, help="CSV only: prediction column name.")
    p.add_argument("--reference-column", type=str, default=None, help="CSV only: reference column name.")
    p.add_argument("--no-bertscore", action="store_true")
    p.add_argument("--bertscore-model", type=str, default="microsoft/deberta-base-mnli")
    p.add_argument("--bertscore-device", type=str, default=None)
    args = p.parse_args()

    preds, refs = load_pairs(args.input, args.prediction_column, args.reference_column)
    result = evaluate(
        preds, refs,
        with_bertscore=not args.no_bertscore,
        bertscore_model=args.bertscore_model,
        bertscore_device=args.bertscore_device,
    )
    result["input_file"] = str(args.input)
    print_summary(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] metrics written to {args.output}")


if __name__ == "__main__":
    main()
