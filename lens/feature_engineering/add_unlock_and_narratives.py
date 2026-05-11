#!/usr/bin/env python3
"""Add the per-minute phone-unlock column and narrative text to feature rows.

This consolidates three steps from the original pipeline (``lock.py`` ->
``merge.py`` -> ``with_narrative.py``):

  1. For each EMA, slice the preceding 4-hour window of the 1 Hz phone-unlock
     stream and down-sample to per-minute "any unlock" (0/1), giving the
     ``unlock_min`` column (length 240 over 4 h).
  2. Left-merge that column onto ``feature_rows.csv`` by ``(uid, response_time)``.
  3. Attach narrative text: the rule-based summary narrative (always recomputed
     from the EMA item columns via :mod:`lens.feature_engineering.ema_codebook`)
     and, if an ``enriched_narratives.jsonl`` is present under the data root, the
     LLM-rewritten ``enhanced_narrative`` joined by ``(uid, ema_timestamp)``.

The result is ``filtered_feature_rows.csv`` -- the input consumed by the
dataset-build stage.

Usage::

    python -m lens.feature_engineering.add_unlock_and_narratives \
        --data-root data/fake \
        --in data/fake/feature_rows.csv --out data/fake/filtered_feature_rows.csv
"""
from __future__ import annotations

import argparse
import ast
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from . import ema_codebook
from ._paths import find_file, sensor_channel_dir
from .build_feature_rows import load_ema_filtered

EMA_ITEM_COLUMNS = [f"ema_{q}" for q in ema_codebook.EMA_ITEMS]


# --------------------------------------------------------------------------- #
# Phone-unlock window -> per-minute 0/1
# --------------------------------------------------------------------------- #
def load_unlock(uid: str, data_root: Path) -> Optional[pd.DataFrame]:
    """Load ``<uid>_unlock.csv``: one row per date with an 86400-element 0/1 list."""
    file_path = find_file(sensor_channel_dir(data_root, "unlock"), f"{uid}_unlock.csv")
    if file_path is None or not file_path.exists():
        return None
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None
    if "date" not in df.columns or "data" not in df.columns:
        return None
    df["date_norm"] = pd.to_datetime(df["date"].astype(str).str.strip(), errors="coerce").dt.normalize()
    df = df.dropna(subset=["date_norm"])

    def _parse_arr(s):
        try:
            a = ast.literal_eval(s) if isinstance(s, str) else s
            a = np.asarray(a, dtype=np.float32)
            if a.ndim != 1 or a.size != 86400:
                return None
            return (a > 0.5).astype(np.uint8)
        except Exception:
            return None

    df["arr"] = df["data"].apply(_parse_arr)
    df = df.dropna(subset=["arr"])
    if df.empty:
        return None
    return df[["date_norm", "arr"]].reset_index(drop=True)


def extract_unlock_window(unlock_df: Optional[pd.DataFrame], end_time: pd.Timestamp,
                          hours: int = 4, downsample_sec: int = 60) -> list:
    if unlock_df is None or unlock_df.empty or pd.isna(end_time):
        return []
    start_time = end_time - timedelta(hours=hours)
    out_chunks = []
    for day0 in pd.date_range(start_time.normalize(), end_time.normalize(), freq="D"):
        row = unlock_df.loc[unlock_df["date_norm"] == day0]
        if row.empty:
            continue
        arr = row.iloc[0]["arr"]
        day_start = pd.Timestamp(day0)
        s = max(start_time, day_start)
        e = min(end_time, day_start + timedelta(days=1))
        if s >= e:
            continue
        i0 = max(0, min(86400, int((s - day_start).total_seconds())))
        i1 = max(0, min(86400, int((e - day_start).total_seconds())))
        if i1 > i0:
            out_chunks.append(arr[i0:i1])
    if not out_chunks:
        return []
    sec_series = np.concatenate(out_chunks, axis=0)
    n = int(np.ceil(sec_series.size / downsample_sec))
    pad = n * downsample_sec - sec_series.size
    if pad:
        sec_series = np.pad(sec_series, (0, pad), mode="constant", constant_values=0)
    minute_blocks = sec_series.reshape(n, downsample_sec).max(axis=1)
    return minute_blocks.astype(int).tolist()


def _unlock_for_uid(uid: str, subdf: pd.DataFrame, data_root: Path,
                    hours: int, downsample_sec: int) -> List[dict]:
    unlock_df = load_unlock(uid, data_root)
    rows = []
    for _, r in subdf.iterrows():
        end_time = r["day_parsed"]
        unlock_min = extract_unlock_window(unlock_df, end_time, hours=hours, downsample_sec=downsample_sec)
        rows.append({
            "uid": r["uid"],
            "response_time": end_time.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(end_time) else "",
            "unlock_min": json.dumps(unlock_min),
            "unlock_min_count": len(unlock_min),
        })
    return rows


def build_unlock_rows(data_root: Path, hours: int = 4, downsample_sec: int = 60,
                      workers: int = 16) -> pd.DataFrame:
    data_root = Path(data_root)
    ema_df = load_ema_filtered(data_root / "ema.csv", data_root / "filtered_uids.json")
    rows: List[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_unlock_for_uid, uid, subdf, data_root, hours, downsample_sec)
                for uid, subdf in ema_df.groupby("uid")]
        for fut in as_completed(futs):
            rows.extend(fut.result())
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Narrative join
# --------------------------------------------------------------------------- #
def load_enriched_narratives(jsonl_path: Path) -> pd.DataFrame:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append({
                "uid": obj.get("uid"),
                "ema_timestamp": obj.get("ema_timestamp"),
                "enhanced_narrative": obj.get("enhanced_narrative", ""),
                "rule_based_narrative_src": obj.get("rule_based_narrative", obj.get("template_narrative", "")),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    dt = pd.to_datetime(df["ema_timestamp"], errors="coerce")
    try:
        dt = dt.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    df = df.assign(_ts=dt).dropna(subset=["_ts"])
    df = df.sort_values(["uid", "_ts"]).drop_duplicates(["uid", "_ts"], keep="first")
    return df


def _rule_based_narrative_for_row(row: pd.Series) -> str:
    scores = {c: row[c] for c in EMA_ITEM_COLUMNS if c in row.index}
    return ema_codebook.summary_narrative(scores)


def add_unlock_and_narratives(data_root: Path, in_csv: Path, out_csv: Path,
                              hours: int = 4, downsample_sec: int = 60,
                              workers: int = 16) -> pd.DataFrame:
    data_root = Path(data_root)
    df = pd.read_csv(in_csv)

    # -- 1/2. unlock per-minute column, left-merged by (uid, response_time) --
    unlock_df = build_unlock_rows(data_root, hours=hours, downsample_sec=downsample_sec, workers=workers)
    if not unlock_df.empty:
        df["_rt"] = pd.to_datetime(df["response_time"], errors="coerce")
        unlock_df["_rt"] = pd.to_datetime(unlock_df["response_time"], errors="coerce")
        df = df.merge(unlock_df[["uid", "_rt", "unlock_min", "unlock_min_count"]],
                      on=["uid", "_rt"], how="left").drop(columns=["_rt"])
    else:
        df["unlock_min"] = json.dumps([])
        df["unlock_min_count"] = 0

    # -- 3. narrative text --
    # Always recompute the rule-based summary narrative from the EMA columns.
    df["rule_based_narrative"] = df.apply(_rule_based_narrative_for_row, axis=1)

    # If the LLM-rewritten narratives are available, join them by (uid, timestamp).
    enriched_path = data_root / "enriched_narratives.jsonl"
    if enriched_path.exists():
        nar = load_enriched_narratives(enriched_path)
        if not nar.empty:
            df["_ts"] = pd.to_datetime(df["response_time"], errors="coerce")
            try:
                df["_ts"] = df["_ts"].dt.tz_localize(None)
            except (TypeError, AttributeError):
                pass
            df = df.merge(nar[["uid", "_ts", "enhanced_narrative"]], how="left",
                          left_on=["uid", "_ts"], right_on=["uid", "_ts"]).drop(columns=["_ts"])
        else:
            df["enhanced_narrative"] = ""
    else:
        df["enhanced_narrative"] = ""
    df["enhanced_narrative"] = df["enhanced_narrative"].fillna("")

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows to {out_csv}")
    return df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add the per-minute unlock column and narrative text to feature rows.")
    parser.add_argument("--data-root", type=str, default="data/fake",
                        help="dataset root (contains ema.csv, raw_sensors/unlock/, enriched_narratives.jsonl)")
    parser.add_argument("--in", dest="in_csv", type=str, default="data/fake/feature_rows.csv",
                        help="input feature_rows.csv")
    parser.add_argument("--out", dest="out_csv", type=str, default="data/fake/filtered_feature_rows.csv")
    parser.add_argument("--hours", type=int, default=4, help="pre-EMA window length (hours)")
    parser.add_argument("--ds", dest="downsample_sec", type=int, default=60,
                        help="unlock down-sample step (seconds)")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    add_unlock_and_narratives(Path(args.data_root), Path(args.in_csv), Path(args.out_csv),
                              hours=args.hours, downsample_sec=args.downsample_sec, workers=args.workers)


if __name__ == "__main__":
    main()
