#!/usr/bin/env python3
"""End-to-end EMA -> rule-based templates -> LLM enrichment pipeline driver.

Stages (paper Section 3.1):

  1. **Templates**  -- :class:`EMATemplateGenerator` turns the feature-rows CSV
     into deterministic item-level + summary-level narrative templates.
  2. **Enrichment** -- an instruction-tuned LLM (Qwen2.5-14B; served locally via
     SGLang) rewrites each template into a fluent narrative.  With ``--mock-llm``
     the LLM call is skipped and a deterministic stub rewrite is substituted
     (``"[mock-enriched] " + template``), so the whole pipeline runs offline.
  3. **Quality control** -- the multi-agent LLM-as-a-judge ensemble
     (:mod:`llm_evaluation`) scores each rewrite; PASS narratives become the
     training labels.  (Step 3 is a separate entry point;
     :func:`run_pipeline` only invokes 1-2.)

Outputs (written under ``data_root``, schema matching ``data/fake/generate.py``):

  * ``enriched_narratives.jsonl`` -- ``{uid, ema_timestamp, rule_based_narrative,
    enhanced_narrative}``
  * ``enriched_qas.jsonl`` -- ``{uid, ema_timestamp, question_key, enhanced_answer}``

Config (YAML): ``data_root``, ``feature_rows`` (optional explicit CSV path),
``model_name``, ``sglang_url``, ``max_records``, ``temperature``, ``seed``.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .template_enrichment.generator import EMATemplateGenerator
from .template_enrichment.prompt import create_enrichment_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MOCK_PREFIX = "[mock-enriched] "


# --------------------------------------------------------------------------- #
# Config / path resolution
# --------------------------------------------------------------------------- #
def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_feature_rows(data_root: Path, cfg: Dict[str, Any], seed: int) -> Path:
    """Find the feature-rows CSV, or build a minimal stand-in from ``ema.csv``.

    The feature-engineering subpackage produces ``filtered_feature_rows.csv``;
    if it has not been run yet we fall back to ``feature_rows.csv``, then to a
    synthesized CSV derived from ``ema.csv`` (with synthetic raw time-series
    columns) so the pipeline still runs standalone.
    """
    explicit = cfg.get("feature_rows")
    if explicit and Path(explicit).exists():
        return Path(explicit)
    for name in ("filtered_feature_rows.csv", "feature_rows.csv"):
        p = data_root / name
        if p.exists():
            return p
    # Synthesize from ema.csv -- reuse the build_dataset fallback for raw TS columns.
    from .dataset_build.build_dataset import _synthesize_feature_rows  # local import

    ema_csv = data_root / "ema.csv"
    if not ema_csv.exists():
        raise SystemExit(f"No feature-rows CSV and no {ema_csv} to synthesize from")
    logger.info("[fallback] No feature-rows CSV found; synthesizing from %s", ema_csv)
    df = _synthesize_feature_rows(str(ema_csv), str(data_root / "enriched_narratives.jsonl"),
                                  str(data_root / "enriched_qas.jsonl"), seed)
    out = data_root / "filtered_feature_rows.csv"
    df.to_csv(out, index=False)
    return out


# --------------------------------------------------------------------------- #
# Mock vs real enrichment
# --------------------------------------------------------------------------- #
def _mock_rewrite(text: str) -> str:
    """Deterministic offline stand-in for the LLM rewrite."""
    return MOCK_PREFIX + (text or "").strip()


def _real_rewrite_batch(texts: List[str], cfg: Dict[str, Any]) -> List[str]:
    """Rewrite a batch of templates via the SGLang/OpenAI-compatible LLM server."""
    from .template_enrichment.sglang_batch_processor import (BatchConfig,  # local import
                                                             SGLangConcurrentProcessor)

    bc = BatchConfig(sglang_url=cfg.get("sglang_url", "http://localhost:30000"),
                     model_name=cfg.get("model_name", "qwen2.5-14b"),
                     max_tokens=int(cfg.get("max_tokens", 256)),
                     temperature=float(cfg.get("temperature", 0.1)),
                     max_workers=int(cfg.get("max_workers", 16)))
    proc = SGLangConcurrentProcessor(bc)
    if not proc.check_server():
        raise SystemExit(f"LLM server not reachable at {bc.sglang_url}; "
                         "start it or pass --mock-llm")
    out: List[str] = []
    for t in texts:
        payload = {"custom_id": "x", "group_id": 0, "task_type": "narrative",
                   "model": bc.model_name,
                   "messages": create_enrichment_prompt(t),
                   "max_tokens": bc.max_tokens, "temperature": bc.temperature,
                   "stop": ["**", "Input:", "Output:", "Example"]}
        res = proc._execute_single_request(payload)  # noqa: SLF001
        out.append((res or {}).get("enhanced_text", "") or t)
    return out


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(config_path: Optional[str], data_root: Optional[str], mock_llm: bool) -> None:
    cfg = _load_config(config_path)
    root = Path(data_root or cfg.get("data_root") or "data/fake")
    root.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 20260511))
    max_records = int(cfg.get("max_records", 0))
    random.seed(seed)

    gen = EMATemplateGenerator()
    csv_path = _resolve_feature_rows(root, cfg, seed)
    logger.info("Stage 1: rule-based templates from %s", csv_path)

    templates_json = root / "template_narratives.json"
    sampled_json = root / "template_sampled_question_responses.json"
    gen.process_csv(str(csv_path), str(templates_json), str(sampled_json),
                    n_sampled=int(cfg.get("n_sampled_qa", 3)), seed=seed)

    with open(templates_json, "r", encoding="utf-8") as f:
        nar_templates = json.load(f)
    with open(sampled_json, "r", encoding="utf-8") as f:
        sampled = json.load(f)
    if max_records > 0:
        nar_templates = nar_templates[:max_records]
        sampled = sampled[:max_records]

    # Flatten the sampled item responses into (uid, ts, qkey, text) triples.
    qa_items = []
    for rec in sampled:
        for qkey, qtext in (rec.get("question_responses") or {}).items():
            qa_items.append((rec.get("uid"), rec.get("ema_timestamp"), qkey, qtext))

    logger.info("Stage 2: %s enrichment (%d narratives, %d item answers)",
                "MOCK" if mock_llm else "LLM", len(nar_templates), len(qa_items))
    if mock_llm:
        nar_outputs = [_mock_rewrite(r.get("template_narrative", "")) for r in nar_templates]
        qa_outputs = [_mock_rewrite(t) for (_, _, _, t) in qa_items]
    else:
        nar_outputs = _real_rewrite_batch([r.get("template_narrative", "") for r in nar_templates], cfg)
        qa_outputs = _real_rewrite_batch([t for (_, _, _, t) in qa_items], cfg)

    nar_out_path = root / "enriched_narratives.jsonl"
    qa_out_path = root / "enriched_qas.jsonl"
    with open(nar_out_path, "w", encoding="utf-8") as f:
        for rec, enriched in zip(nar_templates, nar_outputs):
            f.write(json.dumps({
                "uid": rec.get("uid"),
                "ema_timestamp": rec.get("ema_timestamp"),
                "rule_based_narrative": rec.get("template_narrative", ""),
                "enhanced_narrative": enriched,
            }, ensure_ascii=False) + "\n")
    with open(qa_out_path, "w", encoding="utf-8") as f:
        for (uid, ts, qkey, qtext), enriched in zip(qa_items, qa_outputs):
            f.write(json.dumps({
                "uid": uid, "ema_timestamp": ts, "question_key": qkey,
                "rule_based_answer": qtext, "enhanced_answer": enriched,
            }, ensure_ascii=False) + "\n")

    logger.info("Pipeline complete. Wrote:")
    logger.info("  %s  (%d rows)", nar_out_path, len(nar_templates))
    logger.info("  %s  (%d rows)", qa_out_path, len(qa_items))


def main():
    parser = argparse.ArgumentParser(description="LENS data pipeline: templates -> LLM enrichment")
    parser.add_argument("--config", help="YAML config file")
    parser.add_argument("--data-root", help="Override data_root from config")
    parser.add_argument("--mock-llm", action="store_true",
                        help="Skip all LLM calls; substitute deterministic stub rewrites (offline)")
    args = parser.parse_args()
    run_pipeline(args.config, args.data_root, args.mock_llm)


if __name__ == "__main__":
    main()
