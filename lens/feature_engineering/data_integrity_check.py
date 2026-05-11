#!/usr/bin/env python3
"""Post-merge sanity checks for the per-EMA feature table.

Verifies that each fixed-rate sensor channel has the expected length over the
4-hour pre-EMA window (heart rate 1440, ZCR 480, steps 240, GPS 24) and reports
any rows that disagree.

Usage::

    python -m lens.feature_engineering.data_integrity_check \
        --csv data/fake/filtered_feature_rows.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm import tqdm

# Expected per-channel lengths over a 4-hour window (see paper Appendix C).
EXPECTED_COUNTS: Dict[str, int] = {
    "gps_count": 24,
    "hr_count": 1440,
    "steps_count": 240,
    "zcr_count": 480,
}


def check_data_integrity(csv_path: Path, expected_counts: Dict[str, int] = EXPECTED_COUNTS) -> list:
    print(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")

    # Some legacy tables carry a ``match_success`` flag; honour it if present.
    if "match_success" in df.columns:
        df = df[df["match_success"] == True].copy()  # noqa: E712
        print(f"Rows with match_success = True: {len(df)}")

    print("\n--- Max values of count columns ---")
    for col in expected_counts:
        if col in df.columns:
            clean = pd.to_numeric(df[col], errors="coerce").dropna()
            if not clean.empty:
                print(f"Max value for '{col}': {int(clean.max())}")
            else:
                print(f"Column '{col}' is empty or non-numeric.")
        else:
            print(f"Column '{col}' not found.")
    print("-----------------------------------")

    # The timestamp column has been renamed across pipeline versions.
    ts_col = next((c for c in ("ema_timestamp", "response_time") if c in df.columns), None)

    discrepancies = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking counts"):
        ema_timestamp = row.get(ts_col) if ts_col else None
        uid = row.get("uid")
        for col, expected in expected_counts.items():
            current = row.get(col)
            if current is None or pd.isna(current):
                discrepancies.append({"uid": uid, "ema_timestamp": ema_timestamp,
                                      "column": col, "error": f"Column '{col}' missing or None."})
            elif int(current) != expected:
                discrepancies.append({"uid": uid, "ema_timestamp": ema_timestamp, "column": col,
                                      "expected_count": expected, "actual_count": int(current)})

    if discrepancies:
        print("\n--- Discrepancies found ---")
        for disc in discrepancies:
            print(disc)
        print(f"Total discrepancies: {len(discrepancies)}")
    else:
        print("\n--- No discrepancies found in sensor counts ---")
    print("Data integrity check finished.")
    return discrepancies


def main() -> None:
    parser = argparse.ArgumentParser(description="Check sensor-channel lengths in the feature table.")
    parser.add_argument("--csv", type=str, default="data/fake/filtered_feature_rows.csv",
                        help="path to feature_rows.csv / filtered_feature_rows.csv")
    args = parser.parse_args()
    check_data_integrity(Path(args.csv))


if __name__ == "__main__":
    main()
