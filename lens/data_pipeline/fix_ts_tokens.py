#!/usr/bin/env python3
"""Step (3) of the data-build chain: rewrite ``<ts></ts>`` -> ``<ts><ts/>``.

The ``build_dataset`` -> ``convert_hf_to_jsonl`` stages emit the time-series
placeholder as ``<ts></ts>``.  The ChatTS ``chatts`` chat template tokenizer
that stage-1 / stage-2 SFT uses expects the self-closing form ``<ts><ts/>``
(and may also encounter the JSON-escaped variant ``<\\/ts>``).  This script
fixes every string field of every JSONL line, in place, in the given files or
directories, keeping a ``.backup`` copy of each modified file.

Usage:
    python -m lens.data_pipeline.fix_ts_tokens path/to/dir_or_file [more ...]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Tuple

from tqdm import tqdm


def fix_ts_tokens(text):
    """Rewrite ``</ts>`` (and the escaped ``<\\/ts>``) to ``<ts/>`` in a string."""
    if not isinstance(text, str):
        return text
    return text.replace(r"<\/ts>", "<ts/>").replace("</ts>", "<ts/>")


def process_jsonl_file(file_path: Path) -> Tuple[int, int]:
    """Fix one JSONL file in place. Returns ``(num_replacements, num_parse_errors)``."""
    backup_path = Path(str(file_path) + ".backup")
    shutil.copy2(file_path, backup_path)

    fixed_lines = []
    error_count = 0
    total_replacements = 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line_num, line in enumerate(tqdm(lines, desc=f"  {file_path.name}", leave=False), 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  warning: line {line_num} JSON decode error: {exc}")
                error_count += 1
                fixed_lines.append(line)
                continue
            for key, value in data.items():
                if isinstance(value, str):
                    original = value
                    data[key] = fix_ts_tokens(value)
                    if original != data[key]:
                        total_replacements += original.count(r"<\/ts>") + original.count("</ts>")
            fixed_lines.append(json.dumps(data, ensure_ascii=False))
        with open(file_path, "w", encoding="utf-8") as f:
            for line in fixed_lines:
                f.write(line + "\n")
        print(f"  {file_path}: fixed {total_replacements} placeholders"
              + (f", {error_count} parse errors" if error_count else ""))
        return total_replacements, error_count
    except Exception:
        shutil.copy2(backup_path, file_path)
        raise


def process_path(path: Path) -> Tuple[int, int]:
    """Fix a single JSONL file, or every ``*.jsonl`` under a directory."""
    if path.is_file():
        return process_jsonl_file(path)
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if not files:
            print(f"No .jsonl files found under {path}")
            return 0, 0
        total_r = total_e = 0
        for f in files:
            r, e = process_jsonl_file(f)
            total_r += r
            total_e += e
        return total_r, total_e
    print(f"Path does not exist: {path}")
    return 0, 0


def main():
    parser = argparse.ArgumentParser(description="Rewrite <ts></ts> -> <ts><ts/> in JSONL files")
    parser.add_argument("paths", nargs="+", help="JSONL files or directories of JSONL files")
    args = parser.parse_args()
    grand_r = grand_e = 0
    for p in args.paths:
        r, e = process_path(Path(p))
        grand_r += r
        grand_e += e
    print(f"All done: {grand_r} placeholders fixed, {grand_e} parse errors. Backups: *.backup")


if __name__ == "__main__":
    main()
