#!/usr/bin/env python3
"""Pull the ``enhanced_narrative`` / ``enhanced_answer`` fields out of LLM outputs.

The template-enrichment stage emits JSONL where each line carries a rule-based
template plus an LLM-rewritten field.  This utility strips everything except the
rewritten text, producing slim JSONL files for downstream inspection / dataset
assembly.

Usage::

    python -m lens.feature_engineering.extract_enhanced_fields \
        --narratives data/fake/enriched_narratives.jsonl \
        --qas data/fake/enriched_qas.jsonl \
        --out-dir data/fake/enhanced_extracted
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _extract_field(input_file: Path, output_file: Path, field: str) -> int:
    count = 0
    with open(input_file, "r", encoding="utf-8") as f_in, \
            open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = data.get(field, "")
            if value:
                f_out.write(json.dumps({field: value}, ensure_ascii=False) + "\n")
                count += 1
    return count


def extract_enhanced_narrative(input_file: Path, output_file: Path) -> int:
    """Extract ``enhanced_narrative`` from an enriched-narratives JSONL."""
    n = _extract_field(Path(input_file), Path(output_file), "enhanced_narrative")
    print(f"Extracted {n} enhanced_narrative records to {output_file}")
    return n


def extract_enhanced_answer(input_file: Path, output_file: Path) -> int:
    """Extract ``enhanced_answer`` from an enriched-QAs JSONL."""
    n = _extract_field(Path(input_file), Path(output_file), "enhanced_answer")
    print(f"Extracted {n} enhanced_answer records to {output_file}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract enhanced narrative/answer fields.")
    parser.add_argument("--narratives", type=str, default="data/fake/enriched_narratives.jsonl",
                        help="input enriched-narratives JSONL")
    parser.add_argument("--qas", type=str, default="data/fake/enriched_qas.jsonl",
                        help="input enriched-QAs JSONL")
    parser.add_argument("--out-dir", type=str, default="data/fake/enhanced_extracted")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    narratives_in = Path(args.narratives)
    qas_in = Path(args.qas)
    if narratives_in.exists():
        extract_enhanced_narrative(narratives_in, out_dir / "enhanced_narratives.jsonl")
    if qas_in.exists():
        extract_enhanced_answer(qas_in, out_dir / "enhanced_answers.jsonl")
    print(f"Output written under: {out_dir}")


if __name__ == "__main__":
    main()
