#!/usr/bin/env python3
"""Step (2) of the data-build chain: Arrow ``DatasetDict`` -> JSONL splits.

Reads the ``narrative_dataset`` and ``qa_dataset`` Arrow datasets produced by
:mod:`build_dataset` and writes one ``{split}.jsonl`` per split (train /
validation / test) for each.  The time-series placeholder is still ``<ts></ts>``
at this stage; :mod:`lens.data_pipeline.fix_ts_tokens` rewrites it to the final
``<ts><ts/>`` form afterwards.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_from_disk


def convert_hf_dataset_to_jsonl(dataset_path: Path, output_dir: Path) -> int:
    """Export every split of an Arrow dataset to ``{split}.jsonl`` under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict = load_from_disk(str(dataset_path))
    total = 0
    for split_name in dataset_dict.keys():
        split_ds = dataset_dict[split_name]
        out_path = output_dir / f"{split_name}.jsonl"
        split_ds.to_json(str(out_path), orient="records", lines=True, force_ascii=False)
        print(f"  {dataset_path.name}/{split_name}: {len(split_ds)} -> {out_path}")
        total += len(split_ds)
    return total


def main():
    parser = argparse.ArgumentParser(description="Convert LENS Arrow datasets to JSONL splits")
    parser.add_argument("--root", required=True,
                        help="Directory containing the Arrow datasets (narrative_dataset/, qa_dataset/)")
    parser.add_argument("--out", required=True,
                        help="Output directory root; JSONL splits go under {out}/{dataset_name}/")
    parser.add_argument("--datasets", nargs="+", default=["narrative_dataset", "qa_dataset"],
                        help="Dataset subdirectory names to convert")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    grand_total = 0
    for name in args.datasets:
        in_path = root / name
        if not in_path.exists():
            print(f"[skip] {in_path} does not exist")
            continue
        print(f"Converting {name} ...")
        grand_total += convert_hf_dataset_to_jsonl(in_path, out / name)
    print(f"Done. {grand_total} total samples across {len(args.datasets)} datasets.")


if __name__ == "__main__":
    main()
