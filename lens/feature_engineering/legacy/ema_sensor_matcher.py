#!/usr/bin/env python3
"""Legacy EMA <-> sensor matcher (SUPERSEDED -- kept for reference only).

This is the earlier "preload everything, then match" implementation of the
raw -> feature_rows step.  It has been superseded by
:mod:`lens.feature_engineering.build_feature_rows`, which uses per-channel
``searchsorted`` window extraction and is the canonical path used in the paper.

It is retained here because it documents the original per-minute phone-unlock
down-sample and the sleep / conversation scalar logic.  It is not wired into any
``make`` target and is not part of the smoke test.

Usage (for reference)::

    python -m lens.feature_engineering.legacy.ema_sensor_matcher \
        --base-path data/fake --sensors hr,steps,gps --target 50 --output matched.csv
"""
from __future__ import annotations

import argparse
import ast
import json
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _raw_sensors_dir(base_path: Path) -> Path:
    for cand in ("raw_sensors", "raw_data"):
        if (base_path / cand).is_dir():
            return base_path / cand
    return base_path / "raw_sensors"


class EMASensorMatcher:
    """Match each EMA timestamp to the preceding 4-hour window of sensor data."""

    def __init__(self, base_path: str = "data/fake", enable_preload: bool = True,
                 selected_sensors: Optional[List[str]] = None):
        self.base_path = Path(base_path)
        self.raw_dir = _raw_sensors_dir(self.base_path)
        all_sensor_types = ["gps", "hr", "steps", "zcr", "convo", "unlock", "stress", "sleep"]
        if selected_sensors:
            sel = set(s.strip() for s in selected_sensors)
            self.sensor_types = [s for s in all_sensor_types if s in sel]
        else:
            self.sensor_types = all_sensor_types

        self.ema_columns = ["__name__", "day", "uid", "Q1", "Q10", "Q11(0)",
                            "Q11(1)", "Q11(2)", "Q12", "Q13", "Q14", "Q2",
                            "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9"]
        self.results: List[Dict[str, Any]] = []
        self.match_stats = {
            "total_ema_records": 0,
            "successful_matches": 0,
            "failed_matches": 0,
            "sensor_stats": {s: {"success": 0, "fail": 0} for s in self.sensor_types},
        }
        self.preloaded_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.enable_preload = enable_preload
        if self.enable_preload:
            self.preload_all_sensor_data()

    # ------------------------------------------------------------------ #
    def _sensor_dir(self, sensor_type: str) -> Path:
        if sensor_type == "sleep":
            for cand in (self.base_path / "garmin_health_api" / "sleep",
                         self.raw_dir / "garmin_health_api" / "sleep"):
                if cand.is_dir():
                    return cand
            return self.base_path / "garmin_health_api" / "sleep"
        return self.raw_dir / sensor_type

    def _uid_from_file(self, sensor_type: str, file_path: Path) -> str:
        name = file_path.name
        if sensor_type == "convo" and name.startswith("convo_") and name.endswith(".csv"):
            return name[6:-4]
        if sensor_type == "unlock" and name.endswith("_unlock.csv"):
            return name[:-11]
        if sensor_type == "stress" and name.endswith("_603.csv"):
            return name[:-8]
        if sensor_type == "sleep":
            return file_path.stem
        suffix_mapping = {"gps": "_2.csv", "hr": "_601.csv", "steps": "_604.csv", "zcr": "_618.csv"}
        suffix = suffix_mapping.get(sensor_type, "")
        if suffix and name.endswith(suffix):
            return name[: -len(suffix)]
        return file_path.stem

    def preload_all_sensor_data(self) -> None:
        for sensor_type in self.sensor_types:
            sensor_dir = self._sensor_dir(sensor_type)
            self.preloaded_data.setdefault(sensor_type, {})
            if not sensor_dir.exists():
                continue
            for file_path in sensor_dir.glob("*.csv"):
                uid, df = self._load_single_file(sensor_type, file_path)
                if uid and df is not None:
                    self.preloaded_data[sensor_type][uid] = df

    def _load_single_file(self, sensor_type: str, file_path: Path):
        try:
            uid = self._uid_from_file(sensor_type, file_path)
            df = pd.read_csv(file_path)
            time_col = {"convo": "event_time", "unlock": "date", "sleep": "calendarDate"}.get(
                sensor_type, "day")
            if time_col not in df.columns:
                return uid, None
            df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            if df.empty:
                return uid, None
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["_ts_np"] = df["timestamp"].values.astype("datetime64[ns]")
            if sensor_type in ("unlock", "sleep"):
                df["_date_floored"] = df["timestamp"].dt.floor("D")
                bounds = df["_date_floored"].ne(df["_date_floored"].shift()).to_numpy().nonzero()[0].tolist() + [len(df)]
                by_date = {}
                for s, e in zip(bounds[:-1], bounds[1:]):
                    by_date[df["_date_floored"].iloc[s]] = slice(s, e)
                setattr(df, "_by_date_index", by_date)
            return uid, df
        except Exception:
            return None, None

    # ------------------------------------------------------------------ #
    def load_ema_data(self) -> pd.DataFrame:
        ema_df = pd.read_csv(self.base_path / "ema.csv")
        with open(self.base_path / "filtered_uids.json", "r") as f:
            filtered_uids = json.load(f)
        daily_df = ema_df[ema_df["__name__"].str.startswith("daily", na=False)].copy()
        filtered_df = daily_df[daily_df["uid"].isin(filtered_uids)].copy()
        filtered_df["day_parsed"] = pd.to_datetime(filtered_df["day"], errors="coerce")
        available_cols = [c for c in self.ema_columns if c in filtered_df.columns] + ["day_parsed"]
        filtered_df = filtered_df[available_cols].dropna(subset=["day_parsed"])
        self.match_stats["total_ema_records"] = len(filtered_df)
        return filtered_df

    def load_sensor_data(self, uid: str, sensor_type: str) -> Optional[pd.DataFrame]:
        return self.preloaded_data.get(sensor_type, {}).get(uid)

    # ------------------------------------------------------------------ #
    def extract_sensor_features_in_window(self, sensor_data: pd.DataFrame, sensor_type: str,
                                          start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        if sensor_data is None or len(sensor_data) == 0:
            return {"raw_data": [], "count": 0, "data_available": False}

        if sensor_type == "unlock":
            by_date = getattr(sensor_data, "_by_date_index", {})
            date_slice = by_date.get(pd.Timestamp(end_time.date()))
            if date_slice is None or "data" not in sensor_data.columns:
                return {"raw_data": [], "count": 0, "data_available": False}
            data_values = sensor_data.iloc[date_slice]["data"].tolist()
            if not data_values:
                return {"raw_data": [], "count": 0, "data_available": False}
            try:
                data_list = ast.literal_eval(data_values[0])
            except Exception:
                return {"raw_data": [], "count": 0, "data_available": False}
            day_start = datetime.combine(end_time.date(), datetime.min.time())
            ws = max(0, int((end_time - timedelta(hours=4) - day_start).total_seconds()))
            we = min(86400, int((end_time - day_start).total_seconds()) + 1)
            if not (0 <= ws < we):
                return {"raw_data": [], "count": 0, "data_available": False}
            arr = np.asarray(data_list[ws:we], dtype=float)
            if arr.size == 0:
                return {"raw_data": [], "count": 0, "data_available": False}
            full = (arr.size // 60) * 60
            mins: List[int] = []
            if full > 0:
                mins.extend((arr[:full].reshape(-1, 60) != 0.0).any(axis=1).astype(np.uint8).tolist())
            rem = arr[full:]
            if rem.size > 0:
                mins.append(1 if np.any(rem != 0.0) else 0)
            return {"raw_data": mins, "count": len(mins), "data_available": True}

        if sensor_type == "sleep":
            by_date = getattr(sensor_data, "_by_date_index", {})
            date_slice = by_date.get(pd.Timestamp(end_time.date()) - pd.Timedelta(days=1))
            if date_slice is None or "durationInSeconds" not in sensor_data.columns:
                return {"raw_data": [], "count": 0, "data_available": False}
            max_sleep_seconds = sensor_data.iloc[date_slice]["durationInSeconds"].max()
            return {"raw_data": [max_sleep_seconds / 3600.0], "count": 1, "data_available": True}

        if sensor_type == "convo":
            mask = (sensor_data["timestamp"] >= start_time) & (sensor_data["timestamp"] <= end_time)
            window_data = sensor_data[mask]
            if len(window_data) == 0:
                return {"raw_data": [], "count": 0, "data_available": False}
            total = 0.0
            if "event_data" in window_data.columns:
                for s in window_data["event_data"].tolist():
                    try:
                        d = json.loads(s)
                        st, ed = d.get("CONVERSATION_START"), d.get("CONVERSATION_END")
                        if st is not None and ed is not None:
                            total += (int(ed) - int(st)) / 1000.0
                    except Exception:
                        continue
            return {"raw_data": [total], "count": 1, "data_available": True}

        # fixed-rate channels: select the N most-recent points before end_time
        sensor_frequencies = {"gps": 600, "hr": 10, "steps": 60, "zcr": 30, "stress": 60}
        freq = sensor_frequencies.get(sensor_type, 60)
        expected_points = int((4 * 3600) / freq)
        if "_ts_np" in sensor_data.columns:
            ts_np = sensor_data["_ts_np"].values
            end_np = np.datetime64(end_time, "ns")
            pos = np.searchsorted(ts_np, end_np, side="left")
            if pos == 0:
                closest = 0
            elif pos >= len(ts_np):
                closest = len(ts_np) - 1
            else:
                closest = pos - 1 if abs(end_np - ts_np[pos - 1]) <= abs(ts_np[pos] - end_np) else pos
            selected = sensor_data.iloc[max(0, closest - expected_points + 1): closest + 1]
        else:
            mask = (sensor_data["timestamp"] >= start_time) & (sensor_data["timestamp"] <= end_time)
            selected = sensor_data[mask]
            if len(selected) == 0:
                return {"raw_data": [], "count": 0, "data_available": False}

        raw_data: List[Any] = []
        if "data" in selected.columns:
            raw_data = selected["data"].tolist()
            if sensor_type == "gps":
                parsed = []
                for s in raw_data:
                    try:
                        g = json.loads(s)
                        parsed.append({"LONGITUDE": g.get("LONGITUDE"), "LATITUDE": g.get("LATITUDE")})
                    except Exception:
                        continue
                raw_data = parsed
        return {"raw_data": raw_data, "count": len(raw_data), "data_available": len(raw_data) > 0}

    def match_single_ema_record(self, ema_row: pd.Series, window_hours: int = 4) -> Dict[str, Any]:
        uid = ema_row["uid"]
        end_time = ema_row["day_parsed"]
        start_time = end_time - timedelta(hours=window_hours)
        match_result: Dict[str, Any] = {
            "ema_id": ema_row.name,
            "uid": uid,
            "response_time": end_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        }
        ema_cols = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10",
                    "Q11(0)", "Q11(1)", "Q11(2)", "Q12", "Q13", "Q14"]
        for col in ema_cols:
            match_result[f"ema_{col}"] = ema_row[col] if col in ema_row.index else ""

        successful_sensors = 0
        for sensor_type in self.sensor_types:
            sensor_data = self.load_sensor_data(uid, sensor_type)
            if sensor_data is None:
                match_result[f"{sensor_type}_raw_data"] = []
                match_result[f"{sensor_type}_count"] = 0
                continue
            features = self.extract_sensor_features_in_window(sensor_data, sensor_type, start_time, end_time)
            if features.get("data_available", False):
                raw_data = features.get("raw_data", [])
                if sensor_type == "gps":
                    match_result[f"{sensor_type}_raw_data"] = json.dumps(raw_data)
                elif isinstance(raw_data, list):
                    match_result[f"{sensor_type}_raw_data"] = raw_data
                else:
                    match_result[f"{sensor_type}_raw_data"] = [raw_data]
                match_result[f"{sensor_type}_count"] = features.get("count", 0)
                successful_sensors += 1
            else:
                match_result[f"{sensor_type}_raw_data"] = []
                match_result[f"{sensor_type}_count"] = 0
        match_result["match_success"] = successful_sensors > 0
        match_result["successful_sensors"] = successful_sensors
        return match_result

    def process_all_ema_records(self, limit: Optional[int] = None, max_workers: int = 16) -> List[Dict[str, Any]]:
        ema_df = self.load_ema_data()
        if limit:
            ema_df = ema_df.head(limit)
        rows = [(i, r) for i, (_, r) in enumerate(ema_df.iterrows())]
        results: List[Tuple[int, Dict[str, Any]]] = []
        lock = threading.Lock()

        def _work(item):
            i, r = item
            res = self.match_single_ema_record(r)
            with lock:
                if res["match_success"]:
                    self.match_stats["successful_matches"] += 1
                else:
                    self.match_stats["failed_matches"] += 1
            return i, res

        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
            futs = [ex.submit(_work, item) for item in rows]
            for fut in as_completed(futs):
                results.append(fut.result())
        results.sort(key=lambda x: x[0])
        self.results = [r[1] for r in results]
        return self.results

    def save_results(self, output_file: str = "ema_sensor_matched_data.csv", include_raw_data: bool = False) -> Path:
        output_path = self.base_path / output_file
        flattened = []
        for result in self.results:
            flat: Dict[str, Any] = {}
            for key in ("ema_id", "uid", "response_time", "start_time", "end_time",
                        "match_success", "successful_sensors"):
                flat[key] = result.get(key)
            for k, v in result.items():
                if k.startswith("ema_"):
                    flat[k] = v
            for k, v in result.items():
                if k.endswith("_count"):
                    flat[k] = v
                if include_raw_data and k.endswith("_raw_data"):
                    try:
                        flat[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
                    except Exception:
                        flat[k] = str(v)
            flattened.append(flat)
        pd.DataFrame(flattened).to_csv(output_path, index=False, encoding="utf-8")
        return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy EMA-sensor matcher (reference only).")
    parser.add_argument("--base-path", type=str, default="data/fake")
    parser.add_argument("--sensors", type=str, default="", help="comma-separated sensor list; empty = all")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=str, default="ema_sensor_matched_data.csv")
    parser.add_argument("--include-raw", action="store_true")
    args = parser.parse_args()
    selected = [s.strip() for s in args.sensors.split(",") if s.strip()] or None
    matcher = EMASensorMatcher(base_path=args.base_path, enable_preload=True, selected_sensors=selected)
    matcher.process_all_ema_records(limit=args.limit or None, max_workers=args.max_workers)
    out = matcher.save_results(args.output, include_raw_data=args.include_raw)
    print(f"Wrote {len(matcher.results)} rows to {out}")


if __name__ == "__main__":
    main()
