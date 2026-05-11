#!/usr/bin/env python3
"""Build per-EMA feature rows by joining all passive-sensing streams.

For every EMA self-report this script extracts the preceding 4-hour window of
each sensor stream and writes one row to ``feature_rows.csv``.  Windows are cut
with ``searchsorted`` over the sorted per-channel timestamps, so each channel
ends up at its standardised length (heart rate 1440 @ 10s, ZCR 480 @ 30s,
steps/stress 240 @ 60s, GPS lon/lat 24 @ 10min); conversation length (s) and
sleep duration (h, EMA-day) are scalars.  The phone-unlock per-minute column and
the rule-based narrative text are added afterwards by
``add_unlock_and_narratives``.

This is the canonical raw -> feature_rows builder (it supersedes the older
``legacy.ema_sensor_matcher``).

Output columns::

    uid, response_time,
    ema_Q1 .. ema_Q14, ema_Q11(0), ema_Q11(1), ema_Q11(2),
    hr_window, hr_count,
    zcr_first, zcr_last, zcr_count,
    steps_first, steps_count,
    stress_window, stress_count,
    gps_lat, gps_lon, gps_count,
    convo_duration, sleep_duration

Usage::

    python -m lens.feature_engineering.build_feature_rows \
        --data-root data/fake --output data/fake/feature_rows.csv
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

from ._paths import find_file, sensor_channel_dir, sleep_dir

# EMA columns kept from ema.csv (bare item names, no ``ema_`` prefix).
EMA_COLUMNS = ["__name__", "day", "uid", "Q1", "Q10", "Q11(0)",
               "Q11(1)", "Q11(2)", "Q12", "Q13", "Q14", "Q2",
               "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9"]


# --------------------------------------------------------------------------- #
# EMA loading
# --------------------------------------------------------------------------- #
def load_ema_filtered(ema_csv: Path, filtered_uids_json: Path) -> pd.DataFrame:
    ema_df = pd.read_csv(ema_csv)
    with open(filtered_uids_json, "r") as f:
        filtered_uids: List[str] = json.load(f)

    # Keep only "daily_*" EMA prompts and the filtered participant list.
    daily_df = ema_df[ema_df["__name__"].str.startswith("daily", na=False)].copy()
    df = daily_df[daily_df["uid"].isin(filtered_uids)].copy()

    df["day_parsed"] = pd.to_datetime(
        df["day"].astype(str).str.strip(),
        format="%Y-%m-%d %H:%M:%S.%f",
        errors="coerce",
    )

    keep = [c for c in EMA_COLUMNS if c in df.columns] + ["day_parsed"]
    return df[keep]


# --------------------------------------------------------------------------- #
# Generic loaders / window extractors
# --------------------------------------------------------------------------- #
def _read_day_csv(file_path: Optional[Path], time_col: str) -> Optional[pd.DataFrame]:
    """Read a raw sensor CSV that timestamps rows via ``time_col``."""
    if file_path is None or not file_path.exists():
        return None
    try:
        df = pd.read_csv(file_path)
        if time_col not in df.columns:
            return None
        df["timestamp"] = pd.to_datetime(
            df[time_col].astype(str).str.strip(),
            format="%Y-%m-%d %H:%M:%S.%f",
            errors="coerce",
        )
        df = df.dropna(subset=["timestamp"])
        df = df[(df["timestamp"].dt.year >= 2000) & (df["timestamp"].dt.year <= 2030)]
        if df.empty:
            return None
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception:
        return None


def _window_slice(df: pd.DataFrame, end_time: pd.Timestamp, hours: int) -> pd.DataFrame:
    """Return rows in ``[end_time - hours, end_time)`` via searchsorted."""
    start_time = end_time - timedelta(hours=hours)
    ts_ns = df["timestamp"].to_numpy(dtype="datetime64[ns]").astype("int64")
    start_ns = np.int64(pd.Timestamp(start_time).value)
    end_ns = np.int64(pd.Timestamp(end_time).value)
    i0 = ts_ns.searchsorted(start_ns)
    i1 = ts_ns.searchsorted(end_ns)
    return df.iloc[i0:i1]


# ---- heart rate ----
def load_hr(uid: str, data_root: Path) -> Optional[pd.DataFrame]:
    return _read_day_csv(find_file(sensor_channel_dir(data_root, "hr"), f"{uid}_601.csv"), "day")


def extract_hr_window(hr_df: Optional[pd.DataFrame], end_time: pd.Timestamp, hours: int = 4) -> List:
    if hr_df is None or hr_df.empty or pd.isna(end_time):
        return []
    win = _window_slice(hr_df, end_time, hours)
    return win["data"].tolist() if not win.empty else []


# ---- ZCR / pseudoactigraphy ----
def load_zcr(uid: str, data_root: Path) -> Optional[pd.DataFrame]:
    df = _read_day_csv(find_file(sensor_channel_dir(data_root, "zcr"), f"{uid}_618.csv"), "day")
    if df is not None and "data" not in df.columns:
        return None
    return df


def extract_zcr_window(zcr_df: Optional[pd.DataFrame], end_time: pd.Timestamp, hours: int = 4):
    """Return ``(first_values, last_values)`` parsed from the ``[count, .., energy]`` lists."""
    if zcr_df is None or zcr_df.empty or pd.isna(end_time):
        return [], []
    win = _window_slice(zcr_df, end_time, hours)
    if win.empty:
        return [], []
    first_values, last_values = [], []
    for s in win["data"]:
        try:
            arr = ast.literal_eval(s)
            if isinstance(arr, (list, tuple)) and len(arr) >= 1:
                first_values.append(arr[0])
                last_values.append(arr[-1])
        except Exception:
            continue
    return first_values, last_values


# ---- steps ----
def load_steps(uid: str, data_root: Path) -> Optional[pd.DataFrame]:
    df = _read_day_csv(find_file(sensor_channel_dir(data_root, "steps"), f"{uid}_604.csv"), "day")
    if df is not None and "data" not in df.columns:
        return None
    return df


def extract_steps_window(steps_df: Optional[pd.DataFrame], end_time: pd.Timestamp, hours: int = 4) -> List:
    if steps_df is None or steps_df.empty or pd.isna(end_time):
        return []
    win = _window_slice(steps_df, end_time, hours)
    if win.empty:
        return []
    first_values = []
    for s in win["data"]:
        try:
            arr = ast.literal_eval(s)
            if isinstance(arr, (list, tuple)) and len(arr) >= 1:
                first_values.append(arr[0])
        except Exception:
            continue
    return first_values


# ---- stress ----
def load_stress(uid: str, data_root: Path) -> Optional[pd.DataFrame]:
    df = _read_day_csv(find_file(sensor_channel_dir(data_root, "stress"), f"{uid}_603.csv"), "day")
    if df is not None and "data" not in df.columns:
        return None
    return df


def extract_stress_window(stress_df: Optional[pd.DataFrame], end_time: pd.Timestamp, hours: int = 4) -> List:
    if stress_df is None or stress_df.empty or pd.isna(end_time):
        return []
    win = _window_slice(stress_df, end_time, hours)
    return win["data"].tolist() if not win.empty else []


# ---- GPS ----
def load_gps(uid: str, data_root: Path) -> Optional[pd.DataFrame]:
    df = _read_day_csv(find_file(sensor_channel_dir(data_root, "gps"), f"{uid}_2.csv"), "day")
    if df is not None and "data" not in df.columns:
        return None
    return df


def extract_gps_window(gps_df: Optional[pd.DataFrame], end_time: pd.Timestamp, hours: int = 4):
    """Return ``(lat_list, lon_list)`` parsed from the per-fix JSON dicts."""
    if gps_df is None or gps_df.empty or pd.isna(end_time):
        return [], []
    win = _window_slice(gps_df, end_time, hours)
    if win.empty:
        return [], []
    lat_list, lon_list = [], []
    for s in win["data"]:
        try:
            d = json.loads(s)
            lat, lon = d.get("LATITUDE"), d.get("LONGITUDE")
            if lat is not None and lon is not None:
                lat_list.append(lat)
                lon_list.append(lon)
        except Exception:
            continue
    return lat_list, lon_list


# ---- conversation ----
def load_convo(uid: str, data_root: Path) -> Optional[pd.DataFrame]:
    df = _read_day_csv(find_file(sensor_channel_dir(data_root, "convo"), f"convo_{uid}.csv"), "event_time")
    if df is not None and "event_data" not in df.columns:
        return None
    return df


def extract_convo_window(convo_df: Optional[pd.DataFrame], end_time: pd.Timestamp, hours: int = 4) -> float:
    """Return the total conversation duration (seconds) in the window."""
    if convo_df is None or convo_df.empty or pd.isna(end_time):
        return 0.0
    win = _window_slice(convo_df, end_time, hours)
    if win.empty:
        return 0.0
    total = 0.0
    for s in win["event_data"]:
        try:
            d = json.loads(s)
            start, end = d.get("CONVERSATION_START"), d.get("CONVERSATION_END")
            if start is not None and end is not None:
                dur = (int(end) - int(start)) / 1000.0
                if dur > 0:
                    total += dur
        except Exception:
            continue
    return total


# ---- sleep ----
def load_sleep(uid: str, data_root: Path) -> Optional[pd.DataFrame]:
    file_path = find_file(sleep_dir(data_root), f"{uid}.csv")
    if file_path is None or not file_path.exists():
        return None
    try:
        df = pd.read_csv(file_path)
        if "calendarDate" not in df.columns or "durationInSeconds" not in df.columns:
            return None
        df["date"] = pd.to_datetime(
            df["calendarDate"].astype(str).str.strip(), format="%Y-%m-%d", errors="coerce"
        )
        df = df.dropna(subset=["date"])
        df = df[(df["date"].dt.year >= 2000) & (df["date"].dt.year <= 2030)]
        if df.empty:
            return None
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return None


def extract_sleep_window(sleep_df: Optional[pd.DataFrame], end_time: pd.Timestamp) -> float:
    """Return the max sleep duration (hours, 2 dp) recorded on the EMA day."""
    if sleep_df is None or sleep_df.empty or pd.isna(end_time):
        return 0.0
    day = end_time.normalize()
    win = sleep_df.loc[sleep_df["date"] == day]
    if win.empty:
        return 0.0
    max_sec = pd.to_numeric(win["durationInSeconds"], errors="coerce").fillna(0).max()
    return round(max_sec / 3600.0, 2)


# --------------------------------------------------------------------------- #
# Per-UID processing
# --------------------------------------------------------------------------- #
def process_uid(uid: str, subdf: pd.DataFrame, data_root: Path) -> List[dict]:
    hr_df = load_hr(uid, data_root)
    zcr_df = load_zcr(uid, data_root)
    steps_df = load_steps(uid, data_root)
    stress_df = load_stress(uid, data_root)
    gps_df = load_gps(uid, data_root)
    convo_df = load_convo(uid, data_root)
    sleep_df = load_sleep(uid, data_root)

    rows = []
    for _, r in subdf.iterrows():
        end_time = r["day_parsed"]
        hr_list = extract_hr_window(hr_df, end_time, hours=4)
        zcr_first, zcr_last = extract_zcr_window(zcr_df, end_time, hours=4)
        steps_first = extract_steps_window(steps_df, end_time, hours=4)
        stress_list = extract_stress_window(stress_df, end_time, hours=4)
        lat_list, lon_list = extract_gps_window(gps_df, end_time, hours=4)
        convo_dur = extract_convo_window(convo_df, end_time, hours=4)
        sleep_hours = extract_sleep_window(sleep_df, end_time)

        rows.append({
            "uid": r["uid"],
            "response_time": end_time.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(end_time) else "",
            "ema_Q1": r.get("Q1", ""), "ema_Q2": r.get("Q2", ""), "ema_Q3": r.get("Q3", ""),
            "ema_Q4": r.get("Q4", ""), "ema_Q5": r.get("Q5", ""), "ema_Q6": r.get("Q6", ""),
            "ema_Q7": r.get("Q7", ""), "ema_Q8": r.get("Q8", ""), "ema_Q9": r.get("Q9", ""),
            "ema_Q10": r.get("Q10", ""), "ema_Q11(0)": r.get("Q11(0)", ""),
            "ema_Q11(1)": r.get("Q11(1)", ""), "ema_Q11(2)": r.get("Q11(2)", ""),
            "ema_Q12": r.get("Q12", ""), "ema_Q13": r.get("Q13", ""), "ema_Q14": r.get("Q14", ""),
            "hr_window": json.dumps(hr_list), "hr_count": len(hr_list),
            "zcr_first": json.dumps(zcr_first), "zcr_last": json.dumps(zcr_last),
            "zcr_count": len(zcr_first),
            "steps_first": json.dumps(steps_first), "steps_count": len(steps_first),
            "stress_window": json.dumps(stress_list), "stress_count": len(stress_list),
            "gps_lat": json.dumps(lat_list), "gps_lon": json.dumps(lon_list),
            "gps_count": len(lat_list),
            "convo_duration": convo_dur,
            "sleep_duration": sleep_hours,
        })
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_feature_rows(data_root: Path, output: Path, limit: int = 0,
                       workers: int = 16, test_n: int = 0) -> pd.DataFrame:
    data_root = Path(data_root)
    ema_df = load_ema_filtered(data_root / "ema.csv", data_root / "filtered_uids.json")
    if limit and limit > 0:
        ema_df = ema_df.head(limit)

    rows: List[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(process_uid, uid, subdf, data_root)
                for uid, subdf in ema_df.groupby("uid")]
        for fut in as_completed(futs):
            rows.extend(fut.result())
            if test_n and len(rows) >= test_n:
                break
    if test_n:
        rows = rows[:test_n]

    out_df = pd.DataFrame(rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output, index=False)
    print(f"Wrote {len(out_df)} rows to {output}")
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-EMA sensor feature rows.")
    parser.add_argument("--data-root", type=str, default="data/fake",
                        help="dataset root (contains ema.csv, filtered_uids.json, raw_sensors/)")
    parser.add_argument("--output", type=str, default="data/fake/feature_rows.csv")
    parser.add_argument("--limit", type=int, default=0, help="cap on the number of EMA records")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--test-n", type=int, default=0, help="emit at most N rows (debugging)")
    args = parser.parse_args()
    build_feature_rows(Path(args.data_root), Path(args.output),
                       limit=args.limit, workers=args.workers, test_n=args.test_n)


if __name__ == "__main__":
    main()
