#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight BLEU-4 / ROUGE wrapper over a LLaMA-Factory predictions file.

This is a thin port of LLaMA-Factory's ``scripts/eval_bleu_rouge.py`` (Apache-2.0,
the LlamaFactory team). It scores the ``generated_predictions.jsonl`` files
emitted by ``llamafactory-cli`` (records with ``predict`` and ``label`` fields).
For the richer LENS metric suite (ROUGE-1/2/L P/R/F1, BLEU-1..4, METEOR,
BERTScore) use ``lens.eval.metrics.compute_nlp_metrics``.

Requires the optional ``eval`` extra (jieba, nltk, rouge-chinese) — install with
``pip install -e .[eval]`` plus ``pip install jieba rouge-chinese``.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path


def _load_metric_backends():
    try:
        import jieba  # type: ignore
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore
        from rouge_chinese import Rouge  # type: ignore

        jieba.setLogLevel(logging.CRITICAL)
        jieba.initialize()
        return jieba, SmoothingFunction, sentence_bleu, Rouge
    except ImportError as exc:  # pragma: no cover - optional deps
        raise SystemExit(
            "missing optional deps for eval_bleu_rouge: install jieba, nltk, rouge-chinese "
            "(e.g. `pip install -e .[eval] jieba rouge-chinese`)."
        ) from exc


def compute_metrics(sample: dict, jieba, SmoothingFunction, sentence_bleu, Rouge) -> dict:
    hypothesis = list(jieba.cut(sample["predict"]))
    reference = list(jieba.cut(sample["label"]))
    bleu_score = sentence_bleu(
        [list(sample["label"])],
        list(sample["predict"]),
        smoothing_function=SmoothingFunction().method3,
    )
    if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
        result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    else:
        result = Rouge().get_scores(" ".join(hypothesis), " ".join(reference))[0]
    out = {k: round(v["f"] * 100, 4) for k, v in result.items()}
    out["bleu-4"] = round(bleu_score * 100, 4)
    return out


def main(filename: str, out_json: str = "predictions_score.json") -> None:
    jieba, SmoothingFunction, sentence_bleu, Rouge = _load_metric_backends()
    start = time.time()
    rows = []
    with Path(filename).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    per_sample = [compute_metrics(r, jieba, SmoothingFunction, sentence_bleu, Rouge) for r in rows]
    keys = sorted({k for d in per_sample for k in d})
    averages = {k: sum(d.get(k, 0.0) for d in per_sample) / max(1, len(per_sample)) for k in keys}
    for k in keys:
        print(f"{k}: {averages[k]:.4f}")
    Path(out_json).write_text(json.dumps(averages, indent=4), encoding="utf-8")
    print(f"\nDone in {time.time() - start:.3f}s. Score file: {out_json}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m lens.eval.metrics.eval_bleu_rouge <predictions.jsonl> [out.json]", file=sys.stderr)
        sys.exit(2)
    main(*sys.argv[1:3])
