#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NLP reference-based metrics for LENS narratives / QA answers.

Computes ROUGE-1/2/L (precision / recall / F1), BLEU-1..4, METEOR and
BERTScore (precision / recall / F1) for a set of (prediction, reference)
pairs (point estimates).

The heavy NLP dependencies (``rouge-score``, ``sacrebleu``, ``nltk``,
``bert-score``) are part of the optional ``eval`` extra and are imported
lazily — if a package is missing, the corresponding metric is reported as
"skipped" and the script still exits 0. This keeps the smoke-test path
(``--self-test``) runnable using only the core ``pyproject.toml`` deps.

Examples
--------
    # tiny built-in self-test (CI smoke test)
    python -m lens.eval.metrics.compute_nlp_metrics \
        --pred examples/one_sample_each.jsonl \
        --ref  examples/one_sample_each.jsonl \
        --self-test

    # real evaluation: prediction file and reference file (jsonl, `output` field)
    python -m lens.eval.metrics.compute_nlp_metrics \
        --pred inference_results/narrative_results.jsonl \
        --ref  data/fake/narrative_dataset/test.jsonl \
        --pred-field prediction --ref-field output \
        --output metrics.json

    # single jsonl with both `prediction` and `reference` columns
    python -m lens.eval.metrics.compute_nlp_metrics \
        --pred inference_results/narrative_results.jsonl \
        --pred-field prediction --ref-field reference \
        --output metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ``numpy`` is a core dependency.
import numpy as np


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def preprocess_text(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return " ".join(text.split()).strip()


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


# --------------------------------------------------------------------------- #
# Lazy importers — return None if the optional dependency is not installed.
# --------------------------------------------------------------------------- #
def _try_rouge_scorer():
    try:
        from rouge_score import rouge_scorer  # type: ignore

        return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    except Exception:  # ImportError or any init error
        return None


def _try_bleu_metrics():
    try:
        from sacrebleu.metrics import BLEU  # type: ignore

        return {n: BLEU(max_ngram_order=n, effective_order=True) for n in (1, 2, 3, 4)}
    except Exception:
        return None


def _try_nltk_tokenizer():
    try:
        from nltk.tokenize import word_tokenize, wordpunct_tokenize  # type: ignore

        def tok(text: str) -> List[str]:
            try:
                return word_tokenize(text)
            except LookupError:
                return wordpunct_tokenize(text)

        return tok
    except Exception:
        return None


def _try_nltk_meteor():
    try:
        from nltk.translate.meteor_score import meteor_score as nltk_meteor_score  # type: ignore

        return nltk_meteor_score
    except Exception:
        return None


def _try_bert_score():
    try:
        from bert_score import score as bert_score  # type: ignore

        return bert_score
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Metric computation
# --------------------------------------------------------------------------- #
def _zero_rouge() -> Dict[str, Dict[str, float]]:
    z = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {"rouge1": dict(z), "rouge2": dict(z), "rougeL": dict(z)}


def compute_rouge_bleu(
    predictions: List[str],
    references: List[str],
) -> Tuple[Dict[str, Any], Dict[str, List[float]], List[str]]:
    """Returns (summary, per-sample-arrays, skipped-list)."""
    skipped: List[str] = []
    scorer = _try_rouge_scorer()
    bleu = _try_bleu_metrics()

    summary: Dict[str, Any] = {}
    arrays: Dict[str, List[float]] = {}

    if scorer is None:
        skipped.append("rouge (needs rouge-score)")
        summary.update(_zero_rouge())
    else:
        per = {k: {"precision": [], "recall": [], "f1": []} for k in ("rouge1", "rouge2", "rougeL")}
        for pred, ref in zip(predictions, references):
            s = scorer.score(ref, pred)
            for k in ("rouge1", "rouge2", "rougeL"):
                per[k]["precision"].append(s[k].precision)
                per[k]["recall"].append(s[k].recall)
                per[k]["f1"].append(s[k].fmeasure)
        for k in ("rouge1", "rouge2", "rougeL"):
            summary[k] = {m: _mean(per[k][m]) for m in ("precision", "recall", "f1")}
            for m in ("precision", "recall", "f1"):
                arrays[f"{k}_{m}"] = per[k][m]

    if bleu is None:
        skipped.append("bleu (needs sacrebleu)")
        for n in (1, 2, 3, 4):
            summary[f"bleu{n}"] = 0.0
    else:
        per_bleu: Dict[int, List[float]] = {n: [] for n in (1, 2, 3, 4)}
        for pred, ref in zip(predictions, references):
            for n, metric in bleu.items():
                per_bleu[n].append(metric.sentence_score(pred, [ref]).score / 100.0)
        for n in (1, 2, 3, 4):
            summary[f"bleu{n}"] = _mean(per_bleu[n])
            arrays[f"bleu{n}"] = per_bleu[n]

    return summary, arrays, skipped


def compute_meteor(predictions: List[str], references: List[str]) -> Tuple[Optional[float], List[float], List[str]]:
    tok = _try_nltk_tokenizer()
    meteor = _try_nltk_meteor()
    if tok is None or meteor is None:
        return None, [], ["meteor (needs nltk)"]
    scores = [float(meteor([tok(ref)], tok(pred))) for pred, ref in zip(predictions, references)]
    return _mean(scores), scores, []


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    model_type: str = "microsoft/deberta-base-mnli",
    batch_size: int = 64,
    device: Optional[str] = None,
) -> Tuple[Optional[Dict[str, float]], Dict[str, List[float]], List[str]]:
    bert_score = _try_bert_score()
    if bert_score is None:
        return None, {}, ["bertscore (needs bert-score)"]
    try:
        if device is None:
            try:
                import torch  # type: ignore

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        p, r, f1 = bert_score(
            predictions,
            references,
            model_type=model_type,
            verbose=False,
            device=device,
            batch_size=max(1, min(batch_size, len(predictions))),
        )
        p, r, f1 = p.cpu().tolist(), r.cpu().tolist(), f1.cpu().tolist()
        return (
            {"precision": _mean(p), "recall": _mean(r), "f1": _mean(f1)},
            {"precision": p, "recall": r, "f1": f1},
            [],
        )
    except Exception as exc:  # downloads disabled, OOM, ...
        return None, {}, [f"bertscore (failed: {exc})"]


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield i, json.loads(line)


def load_pairs(
    pred_path: Path,
    ref_path: Optional[Path],
    pred_field: str,
    ref_field: str,
) -> Tuple[List[str], List[str]]:
    predictions: List[str] = []
    references: List[str] = []
    if ref_path is None or ref_path.resolve() == pred_path.resolve():
        # single file: both fields in the same records
        for i, rec in _iter_jsonl(pred_path):
            p = rec.get(pred_field)
            r = rec.get(ref_field)
            if p is None:
                # fall back to common alternatives
                p = rec.get("prediction") or rec.get("predict") or rec.get("text")
            if r is None:
                r = rec.get("reference") or rec.get("output") or rec.get("label") or rec.get("ground_truth")
            if p is None or r is None:
                raise ValueError(f"{pred_path}:{i} missing fields '{pred_field}'/'{ref_field}'")
            predictions.append(preprocess_text(p))
            references.append(preprocess_text(r))
    else:
        pred_recs = [rec for _, rec in _iter_jsonl(pred_path)]
        ref_recs = [rec for _, rec in _iter_jsonl(ref_path)]
        if len(pred_recs) != len(ref_recs):
            raise ValueError(
                f"prediction file has {len(pred_recs)} rows but reference file has {len(ref_recs)} rows"
            )
        for i, (pr, rr) in enumerate(zip(pred_recs, ref_recs), start=1):
            p = pr.get(pred_field) or pr.get("prediction") or pr.get("predict") or pr.get("text")
            r = rr.get(ref_field) or rr.get("output") or rr.get("reference") or rr.get("label")
            if p is None or r is None:
                raise ValueError(f"row {i}: missing fields (pred='{pred_field}', ref='{ref_field}')")
            predictions.append(preprocess_text(p))
            references.append(preprocess_text(r))
    if not predictions:
        raise ValueError("no (prediction, reference) pairs found")
    return predictions, references


# --------------------------------------------------------------------------- #
# Evaluation driver
# --------------------------------------------------------------------------- #
def evaluate(
    predictions: List[str],
    references: List[str],
    *,
    with_bertscore: bool = True,
    bertscore_model: str = "microsoft/deberta-base-mnli",
    bertscore_batch_size: int = 64,
    bertscore_device: Optional[str] = None,
) -> Dict[str, Any]:
    summary, _arrays, skipped = compute_rouge_bleu(predictions, references)
    meteor_value, _meteor_scores, meteor_skipped = compute_meteor(predictions, references)
    skipped.extend(meteor_skipped)
    summary["meteor"] = meteor_value if meteor_value is not None else None

    if with_bertscore:
        bs, _bertscore_arrays, bs_skipped = compute_bertscore(
            predictions, references, bertscore_model, bertscore_batch_size, bertscore_device
        )
        skipped.extend(bs_skipped)
        summary["bertscore_precision"] = bs["precision"] if bs else None
        summary["bertscore_recall"] = bs["recall"] if bs else None
        summary["bertscore_f1"] = bs["f1"] if bs else None
    else:
        skipped.append("bertscore (disabled with --no-bertscore)")
        summary["bertscore_precision"] = summary["bertscore_recall"] = summary["bertscore_f1"] = None

    return {
        "total_samples": len(predictions),
        "metrics": summary,
        "skipped_metrics": skipped,
    }


def print_summary(result: Dict[str, Any]) -> None:
    m = result["metrics"]
    print("=" * 60)
    print(f"samples: {result['total_samples']}")
    for k in ("rouge1", "rouge2", "rougeL"):
        v = m.get(k, {})
        print(f"{k:8s}  P={v.get('precision', 0):.4f}  R={v.get('recall', 0):.4f}  F1={v.get('f1', 0):.4f}")
    for n in (1, 2, 3, 4):
        print(f"BLEU-{n}    {m.get(f'bleu{n}', 0):.4f}")
    meteor = m.get("meteor")
    print(f"METEOR    {meteor:.4f}" if meteor is not None else "METEOR    (skipped)")
    bf = m.get("bertscore_f1")
    print(f"BERTScore-F1  {bf:.4f}" if bf is not None else "BERTScore-F1  (skipped)")
    if result.get("skipped_metrics"):
        print("skipped:", "; ".join(result["skipped_metrics"]))
    print("=" * 60)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def run_self_test() -> int:
    """Tiny built-in pred==ref check. Always exits 0 (degrades gracefully)."""
    preds = [
        "The user occasionally felt down and had low energy over the past four hours.",
        "Concentration was difficult and the user reported mild anxiety.",
    ]
    refs = list(preds)  # identical -> ROUGE/BLEU/METEOR should be ~1.0
    result = evaluate(preds, refs, with_bertscore=False)
    print("[self-test] compute_nlp_metrics on a trivial pred==ref pair:")
    print_summary(result)
    m = result["metrics"]
    r1 = m.get("rouge1", {}).get("f1", 0.0)
    if "rouge (needs rouge-score)" in result.get("skipped_metrics", []):
        print("[self-test] rouge-score not installed; ROUGE skipped (expected when 'eval' extra is absent).")
    else:
        # Sanity guard: identical strings must score ~1.0.
        assert r1 > 0.99, f"expected ROUGE-1 F1 ~1.0 for pred==ref, got {r1}"
        print(f"[self-test] OK: ROUGE-1 F1 = {r1:.4f} (>0.99)")
    print("[self-test] PASS")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reference-based NLP metrics for LENS outputs.")
    p.add_argument("--pred", type=Path, help="JSONL with predictions (and optionally references).")
    p.add_argument("--ref", type=Path, default=None,
                   help="JSONL with references (one per line, aligned with --pred). "
                        "Omit to read references from the --pred file itself.")
    p.add_argument("--pred-field", type=str, default="prediction", help="Field name for predictions.")
    p.add_argument("--ref-field", type=str, default="output", help="Field name for references.")
    p.add_argument("--output", type=Path, default=None, help="Optional path to write the metrics JSON.")
    p.add_argument("--no-bertscore", action="store_true", help="Skip BERTScore (avoids a model download).")
    p.add_argument("--bertscore-model", type=str, default="microsoft/deberta-base-mnli")
    p.add_argument("--bertscore-batch-size", type=int, default=64)
    p.add_argument("--bertscore-device", type=str, default=None, help="e.g. 'cpu', 'cuda', 'cuda:0'.")
    p.add_argument("--self-test", action="store_true",
                   help="Run a tiny built-in pred==ref check and exit 0 (CI smoke test).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()

    if args.pred is None:
        print("error: --pred is required (or pass --self-test).", file=sys.stderr)
        return 2

    predictions, references = load_pairs(args.pred, args.ref, args.pred_field, args.ref_field)
    result = evaluate(
        predictions,
        references,
        with_bertscore=not args.no_bertscore,
        bertscore_model=args.bertscore_model,
        bertscore_batch_size=args.bertscore_batch_size,
        bertscore_device=args.bertscore_device,
    )
    result["pred_file"] = str(args.pred)
    if args.ref is not None:
        result["ref_file"] = str(args.ref)
    print_summary(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] metrics written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
