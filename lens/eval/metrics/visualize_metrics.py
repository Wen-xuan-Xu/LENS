#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render Markdown comparison tables from per-model metrics JSON files.

Each metrics JSON is the output of ``lens.eval.metrics.compute_nlp_metrics``
(``{"metrics": {...}}`` with optional ``*_ci`` keys). Given a config that maps
group names to (display name, metrics-json-path) pairs, this writes one
``<name>.md`` table per group with ROUGE-1/2/L F1, BLEU-4, METEOR and
BERTScore-F1 (plus CIs when present).

This is a generalized port of ``symp2textfrom1/visualize_metrics.py`` with all
hardcoded ``/scratch`` paths replaced by a JSON/CLI-driven config.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

METRICS_DEF: List[Tuple[str, str, Optional[str]]] = [
    ("ROUGE-1", "rouge1", "f1"),
    ("ROUGE-2", "rouge2", "f1"),
    ("ROUGE-L", "rougeL", "f1"),
    ("BLEU-4", "bleu4", None),
    ("METEOR", "meteor", None),
    ("BERTScore-F1", "bertscore_f1", None),
]


def load_metrics(path: str) -> Optional[Dict]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def fmt_value(metrics: Optional[Dict], name: str, sub: Optional[str]) -> str:
    if metrics is None:
        return "N/A"
    data = metrics.get("metrics", metrics)
    if sub:
        obj = data.get(name)
        if not isinstance(obj, dict):
            return "N/A"
        val, ci = obj.get(sub), obj.get(f"{sub}_ci")
    else:
        val, ci = data.get(name), data.get(f"{name}_ci")
    if val is None:
        return "N/A"
    if ci and len(ci) == 2 and ci[0] == ci[0]:  # not NaN
        return f"{val:.4f} <br><small>[{ci[0]:.4f}, {ci[1]:.4f}]</small>"
    return f"{val:.4f}"


def render_group(models: List[Tuple[str, str]]) -> str:
    header = "| Model | " + " | ".join(m[0] for m in METRICS_DEF) + " |"
    sep = "|:---| " + " | ".join([":---:"] * len(METRICS_DEF)) + " |"
    lines = [header, sep]
    for display, path in models:
        m = load_metrics(path)
        cells = [fmt_value(m, name, sub) for _, name, sub in METRICS_DEF]
        lines.append(f"| **{display}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Render Markdown metric-comparison tables.")
    p.add_argument("--config", type=Path, required=True,
                   help='JSON: {"Group name": [["Display", "path/to/metrics.json"], ...], ...}')
    p.add_argument("--out", type=Path, default=Path("metrics_summary.md"))
    p.add_argument("--title", type=str, default="Evaluation Summary")
    args = p.parse_args()

    groups: Dict[str, List[Tuple[str, str]]] = {
        g: [tuple(item) for item in models] for g, models in json.loads(args.config.read_text("utf-8")).items()
    }
    out: List[str] = [f"# {args.title}\n"]
    for g, models in groups.items():
        out.append(f"## {g}")
        out.append(render_group(models))
        out.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out), encoding="utf-8")
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
