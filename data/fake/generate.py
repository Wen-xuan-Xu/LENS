#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic synthetic ("fake") data generator for LENS.

The real LENS study data (passive sensing + EMA) is collected under an NIH/NIMH
award and is governed by IRB / funder data-use restrictions, so it is NOT
distributed with this repository. This script fabricates schema-faithful data
that is sufficient to run the whole LENS pipeline end-to-end on a laptop:

  raw sensor CSVs  (data/fake/raw_sensors/, data/fake/garmin_health_api/sleep/)
  ema.csv, filtered_uids.json, Questions.json
  enriched_narratives.jsonl, enriched_qas.jsonl   (stand-ins for LLM outputs)
  narrative_dataset/, qa_dataset/, align_random/, ift/   (final training jsonl)
  dataset_info.json

Everything is derived from a single pinned seed (see data/fake/SEED), so two
runs produce byte-identical output. The fake data is intentionally bland (no
clinical extremes) and all participant IDs are obviously synthetic.

Sampling rates match the paper (Appendix C), over a 4-hour pre-EMA window:
  heart rate  every 10 s  -> length 1440
  ZCR         every 30 s  -> 480  (pseudoactigraphy = zcr_count * energy)
  steps       every 60 s  -> 240
  stress      every 60 s  -> 240
  GPS lon/lat every 10 min-> 24 each
  phone unlock every 60 s -> 240
  sleep (h), conversation (s): scalars per window.

Usage:
  python data/fake/generate.py --out data/fake --all
"""
from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEED_DEFAULT = 20260511
N_USERS = 4                      # fake participants
N_DAYS = 3                       # consecutive study days per participant
EMA_HOURS = (10, 15, 20)         # 3 EMAs/day: morning / afternoon / evening
WINDOW_HOURS = 4                 # pre-EMA sensor window
BASE_DATE = datetime(2026, 5, 4)  # arbitrary, all in the future, obviously synthetic
# A public landmark (Dartmouth Green, Hanover NH) so the fake GPS is plainly fake.
GPS_CENTER_LAT, GPS_CENTER_LON = 43.7044, -72.2887

# Only the channels the LENS pipeline actually consumes (see build_feature_rows /
# add_unlock_and_narratives): heart rate, ZCR, steps, stress, GPS, conversation,
# phone unlock, sleep. The real study collects more (activity / respiration /
# SpO2 / other Garmin Health API streams) but LENS does not use them, so the
# synthetic data omits them.
SENSOR_SUFFIX = {
    "hr": "_601",
    "stress": "_603",
    "steps": "_604",
    "zcr": "_618",
    "gps": "_2",
}

EMA_QUESTIONS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10",
                 "Q11(0)", "Q11(1)", "Q11(2)", "Q12", "Q13"]  # Q14 handled separately (Yes/No)
EMA_NAME_BY_HOUR = {10: "daily_weekday_morning", 15: "daily_weekday_afternoon",
                    20: "daily_weekday_evening"}

# Final per-sample timeseries order (7 streams; matches the 7 <ts><ts/> placeholders).
FINAL_TS_LENGTHS = [1440, 480, 240, 240, 24, 24, 240]  # hr, zcr_prod, steps, stress, gps_lon, gps_lat, unlock


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _uid(i: int) -> str:
    return f"fake_{i:04d}@demo"


def _ema_times() -> Dict[str, List[datetime]]:
    """uid -> sorted list of EMA datetimes."""
    out: Dict[str, List[datetime]] = {}
    for i in range(1, N_USERS + 1):
        uid = _uid(i)
        # offset each user's study window so dates differ a little
        start = BASE_DATE + timedelta(days=3 * (i - 1))
        times = []
        for d in range(N_DAYS):
            day = start + timedelta(days=d)
            for h in EMA_HOURS:
                # jitter a few minutes
                rng = random.Random(hash((uid, d, h)) & 0xFFFFFFFF)
                times.append(day.replace(hour=h, minute=rng.randint(0, 30), second=rng.randint(0, 59)))
        out[uid] = sorted(times)
    return out


def _ts_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _windows(ema_times: List[datetime]) -> List[tuple]:
    """Return (start, end) windows covering [T - WINDOW_HOURS - 12min, T] for each EMA, merged."""
    spans = []
    for t in ema_times:
        spans.append((t - timedelta(hours=WINDOW_HOURS, minutes=12), t))
    spans.sort()
    merged = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _hr_value(dt: datetime, rng: random.Random) -> int:
    # circadian-ish sinusoid + small noise; clamp 50..120
    frac = (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400.0
    base = 70 + 12 * math.sin(2 * math.pi * (frac - 0.25))
    return int(max(50, min(120, base + rng.gauss(0, 4))))


def _stress_value(dt: datetime, rng: random.Random) -> int:
    if rng.random() < 0.06:
        return 4294967294  # Garmin "no measurement" sentinel
    frac = (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400.0
    base = 35 + 20 * math.sin(2 * math.pi * (frac - 0.2))
    return int(max(0, min(100, base + rng.gauss(0, 8))))


# --------------------------------------------------------------------------- #
# Raw-sensor CSV writers (each writes "<idx>,<data>,<day>,<uid>" rows etc.)
# --------------------------------------------------------------------------- #
def _write_indexed_csv(path: Path, header: str, rows: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for idx, r in enumerate(rows):
            f.write(f"{idx},{r}\n")


def gen_raw_sensors(out: Path, ema_times: Dict[str, List[datetime]]) -> None:
    raw = out / "raw_sensors"
    sleep_dir = out / "garmin_health_api" / "sleep"
    for ui, (uid, times) in enumerate(ema_times.items(), start=1):
        rng = random.Random(hash(uid) & 0xFFFFFFFF)
        windows = _windows(times)

        # ---- hr (10 s) ----
        hr_rows = []
        for s, e in windows:
            t = s
            while t <= e:
                hr_rows.append(f"{_hr_value(t, rng)},{_ts_str(t)},{uid}")
                t += timedelta(seconds=10)
        _write_indexed_csv(raw / "hr" / f"{uid}{SENSOR_SUFFIX['hr']}.csv", ",data,day,uid", hr_rows)

        # ---- zcr (30 s): data = "[zcr_count, secs_elapsed, energy]" ----
        zcr_rows = []
        for s, e in windows:
            t = s
            while t <= e:
                cnt = int(max(0, rng.gauss(250, 90)))
                energy = int(max(0, rng.gauss(600, 250)))
                zcr_rows.append(f'"[{cnt},30,{energy}]",{_ts_str(t)},{uid}')
                t += timedelta(seconds=30)
        _write_indexed_csv(raw / "zcr" / f"{uid}{SENSOR_SUFFIX['zcr']}.csv", ",data,day,uid", zcr_rows)

        # ---- steps (60 s buckets in raw): data = "[walk_dur, interval, total_steps, epoch]" ----
        step_rows = []
        for s, e in windows:
            t = s
            total = rng.randint(1000, 4000)
            while t <= e:
                walk = rng.choice([0, 0, 0, 15, 30, 45, 60])
                total += rng.randint(0, 80) if walk else 0
                epoch = int(t.timestamp())
                step_rows.append(f'"[{walk},60,{total},{epoch}]",{_ts_str(t)},{uid}')
                t += timedelta(seconds=60)
        _write_indexed_csv(raw / "steps" / f"{uid}{SENSOR_SUFFIX['steps']}.csv", ",data,day,uid", step_rows)

        # ---- stress (60 s) ----
        st_rows = []
        for s, e in windows:
            t = s
            while t <= e:
                st_rows.append(f"{_stress_value(t, rng)},{_ts_str(t)},{uid}")
                t += timedelta(seconds=60)
        _write_indexed_csv(raw / "stress" / f"{uid}{SENSOR_SUFFIX['stress']}.csv", ",data,day,uid", st_rows)

        # ---- gps (10 min): data = JSON dict ----
        gps_rows = []
        for s, e in windows:
            t = s
            while t <= e:
                lat = round(GPS_CENTER_LAT + rng.gauss(0, 0.002), 7)
                lon = round(GPS_CENTER_LON + rng.gauss(0, 0.002), 7)
                d = {"HASALTITUDE": True, "ALTITUDE": round(150 + rng.gauss(0, 5), 6),
                     "HASACCURACY": True, "LONGITUDE": lon, "ACCURACY": round(rng.uniform(3, 8), 3),
                     "LATITUDE": lat, "PROVIDER": "fused", "STAELLITES": 0,
                     "SPEED": round(max(0.0, rng.gauss(0.3, 0.4)), 6),
                     "BEARING": round(rng.uniform(0, 360), 6), "HASBEARING": True, "HASSPEED": True}
                gps_rows.append(f'"{json.dumps(d).replace(chr(34), chr(34) + chr(34))}",{_ts_str(t)},{uid}')
                t += timedelta(minutes=10)
        _write_indexed_csv(raw / "gps" / f"{uid}{SENSOR_SUFFIX['gps']}.csv", ",data,day,uid", gps_rows)

        # ---- convo: events with JSON {CONVERSATION_START, CONVERSATION_END} (epoch ms) ----
        convo_path = raw / "convo" / f"convo_{uid}.csv"
        convo_path.parent.mkdir(parents=True, exist_ok=True)
        with open(convo_path, "w", encoding="utf-8") as f:
            f.write(",_id,event_data,event_source,event_time,uid\n")
            idx = 0
            for s, e in windows:
                t = s
                while t < e:
                    if rng.random() < 0.3:
                        dur_ms = rng.randint(60_000, 900_000)
                        start_ms = int(t.timestamp() * 1000)
                        end_ms = start_ms + dur_ms
                        ev = json.dumps({"CONVERSATION_START": start_ms, "CONVERSATION_END": end_ms})
                        ev = ev.replace('"', '""')
                        f.write(f'{idx},fakeid{idx:08d},"{ev}",113,{_ts_str(t)},{uid}\n')
                        idx += 1
                    t += timedelta(minutes=12)

        # ---- unlock: one row per EMA date, data = list of 86400 per-second 0/1 ----
        unlock_path = raw / "unlock" / f"{uid}_unlock.csv"
        unlock_path.parent.mkdir(parents=True, exist_ok=True)
        ema_dates = sorted({t.date() for t in times})
        with open(unlock_path, "w", encoding="utf-8") as f:
            f.write(",uid,date,data\n")
            for j, dd in enumerate(ema_dates):
                # ~0.5% of seconds are "unlocked"; bunch them up a bit
                arr = np.zeros(86400, dtype=np.uint8)
                drng = np.random.default_rng(hash((uid, str(dd))) & 0xFFFFFFFF)
                for _ in range(drng.integers(150, 400)):
                    start = int(drng.integers(0, 86340))
                    length = int(drng.integers(1, 60))
                    arr[start:start + length] = 1
                data_str = "[" + ", ".join(f"{float(x)}" for x in arr.tolist()) + "]"
                f.write(f'{j},{uid},{dd.isoformat()},"{data_str}"\n')

        # ---- garmin_health_api/sleep/<uid>.csv ----
        sleep_dir.mkdir(parents=True, exist_ok=True)
        sl_path = sleep_dir / f"{uid}.csv"
        all_dates = sorted({(t - timedelta(days=1)).date() for t in times} | {t.date() for t in times})
        with open(sl_path, "w", encoding="utf-8") as f:
            f.write("_id,calendarDate,durationInSeconds,deepSleepDurationInSeconds,"
                    "lightSleepDurationInSeconds,remSleepInSeconds,awakeDurationInSeconds,mlife_id\n")
            for j, dd in enumerate(all_dates):
                drng = random.Random(hash((uid, "sleep", str(dd))) & 0xFFFFFFFF)
                total = drng.randint(5 * 3600, 9 * 3600)
                deep = int(total * drng.uniform(0.1, 0.2))
                rem = int(total * drng.uniform(0.15, 0.25))
                awake = drng.randint(60, 1200)
                light = max(0, total - deep - rem)
                f.write(f"fakesleep{j:06d},{dd.isoformat()},{total},{deep},{light},{rem},{awake},{uid}\n")


# --------------------------------------------------------------------------- #
# EMA + uid list
# --------------------------------------------------------------------------- #
def gen_ema(out: Path, ema_times: Dict[str, List[datetime]]) -> None:
    rows = []
    for uid, times in ema_times.items():
        for t in times:
            rng = random.Random(hash((uid, _ts_str(t))) & 0xFFFFFFFF)
            row = {"__name__": EMA_NAME_BY_HOUR.get(t.hour, "daily_weekday_morning"),
                   "day": _ts_str(t), "uid": uid}
            for q in EMA_QUESTIONS:
                # bias toward the low end so the synthetic cohort skews "no/minimal"–"mild"
                row[q] = float(min(100, max(0, int(rng.betavariate(1.6, 3.0) * 100))))
            row["Q14"] = "Yes" if rng.random() < 0.15 else "No"
            row["_synthetic"] = "FAKE DATA - not real participant data"
            rows.append(row)
    # write CSV with a stable column order
    cols = ["__name__", "day", "uid"] + EMA_QUESTIONS + ["Q14", "_synthetic"]
    out.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out / "ema.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[""] + cols)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"": i, **r})
    with open(out / "filtered_uids.json", "w", encoding="utf-8") as f:
        json.dump(list(ema_times.keys()), f, indent=2)


# --------------------------------------------------------------------------- #
# Questions.json (small, real-style question bank for dataset_build)
# --------------------------------------------------------------------------- #
QUESTION_BANK = [
    {"question_number": 1, "variants": [
        "In the past 4 hours, how much has the user shown little interest or pleasure in activities?",
        "Rate how little interest or pleasure the user had in doing things over the last 4 hours.",
    ]},
    {"question_number": 2, "variants": [
        "In the past 4 hours, how much has the user appeared down, depressed, or hopeless?",
        "How down, depressed, or hopeless has the user seemed in the last 4 hours?",
    ]},
    {"question_number": 3, "variants": [
        "Last night, how much trouble did the user have with sleep?",
        "Rate the user's sleep disturbance last night.",
    ]},
    {"question_number": 4, "variants": [
        "In the past 4 hours, how tired or low in energy has the user been?",
        "How fatigued or low-energy has the user been over the last 4 hours?",
    ]},
    {"question_number": 5, "variants": [
        "In the past 4 hours, how much has the user shown a poor appetite or overeating?",
        "Rate the user's appetite change in the last 4 hours.",
    ]},
    {"question_number": 6, "variants": [
        "In the past 4 hours, how much has the user felt bad about themselves?",
        "How much self-worth or guilt-related distress has the user shown in the last 4 hours?",
    ]},
    {"question_number": 7, "variants": [
        "In the past 4 hours, how much trouble has the user had concentrating?",
        "Rate the user's difficulty concentrating over the last 4 hours.",
    ]},
    {"question_number": 8, "variants": [
        "In the past 4 hours, how much has the user been moving or speaking more slowly than usual?",
        "Rate any psychomotor change (slowing or restlessness) in the last 4 hours.",
    ]},
    {"question_number": 9, "variants": [
        "In the past 4 hours, how often has the user had thoughts of harming themselves or wishing to be dead?",
        "Rate the presence of suicidal ideation in the last 4 hours.",
    ]},
    {"question_number": 10, "variants": [
        "In the past 4 hours, how much has the user experienced headache, abdominal discomfort, or body aches?",
        "Rate the user's somatic discomfort over the last 4 hours.",
    ]},
    {"question_number": "11(0)", "variants": [
        "In the past 4 hours, how much interest or pleasure has the user had in doing things?",
    ]},
    {"question_number": "11(1)", "variants": [
        "In the past 4 hours, how much energy has the user had?",
    ]},
    {"question_number": "11(2)", "variants": [
        "In the past 4 hours, how well has the user been able to concentrate?",
    ]},
    {"question_number": 12, "variants": [
        "In the past 4 hours, how much has the user felt nervous, anxious, or on edge?",
        "Rate the user's anxiety arousal over the last 4 hours.",
    ]},
    {"question_number": 13, "variants": [
        "In the past 4 hours, how much has the user been unable to stop or control worrying?",
        "Rate how uncontrollable the user's worrying has been in the last 4 hours.",
    ]},
    {"question_number": "overall", "variants": [
        "Please summarize the user's overall mental and physical state in the past 4 hours, "
        "integrating mood, energy, sleep, appetite, concentration, and physical symptoms.",
        "Provide a comprehensive narrative of the user's overall status over the last 4 hours, "
        "combining mood, behavior, and somatic indicators.",
    ]},
]


def gen_questions(out: Path) -> None:
    with open(out / "Questions.json", "w", encoding="utf-8") as f:
        json.dump(QUESTION_BANK, f, indent=2)


# --------------------------------------------------------------------------- #
# enriched_narratives.jsonl / enriched_qas.jsonl (stand-ins for LLM outputs)
# --------------------------------------------------------------------------- #
_FREQ = {(0, 25): "not at all", (26, 50): "sometimes", (51, 75): "often", (76, 100): "constantly"}
_ITEM_TEXT = {
    "Q1": "had little interest or pleasure in doing things",
    "Q2": "felt down, depressed, or hopeless",
    "Q3": "had trouble with sleep last night",
    "Q4": "felt tired or had little energy",
    "Q5": "had a poor appetite or was overeating",
    "Q6": "felt bad about themselves",
    "Q7": "had trouble concentrating",
    "Q8": "moved or spoke slowly, or was fidgeting more",
    "Q9": "had thoughts of self-harm or being better off dead",
    "Q10": "had headaches, abdominal discomfort, or body aches",
    "Q11(0)": "had interest or pleasure in doing things",
    "Q11(1)": "had a lot of energy",
    "Q11(2)": "was able to concentrate well",
    "Q12": "felt nervous, anxious, or on edge",
    "Q13": "was unable to stop or control worrying",
}


def _freq(score: float) -> str:
    for (lo, hi), w in _FREQ.items():
        if lo <= score <= hi:
            return w
    return "not at all"


def gen_enriched(out: Path, ema_times: Dict[str, List[datetime]]) -> None:
    import csv as _csv
    ema = {}
    with open(out / "ema.csv", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            ema[(r["uid"], r["day"])] = r

    nar_path = out / "enriched_narratives.jsonl"
    qa_path = out / "enriched_qas.jsonl"
    with open(nar_path, "w", encoding="utf-8") as fn, open(qa_path, "w", encoding="utf-8") as fq:
        for uid, times in ema_times.items():
            for t in times:
                ts = _ts_str(t)
                row = ema[(uid, ts)]
                # ema_timestamp in dataset_build is matched to feature_rows.response_time,
                # which is "%Y-%m-%d %H:%M:%S" (no microseconds) — use that form here.
                ema_ts = t.strftime("%Y-%m-%d %H:%M:%S")
                # ----- summary narrative -----
                parts = []
                for q in ["Q1", "Q2", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q12", "Q13"]:
                    sc = float(row[q]) if row.get(q) not in (None, "") else 0.0
                    parts.append(f"The user {_freq(sc)} {_ITEM_TEXT[q]}.")
                rule = " ".join(parts)
                enhanced = ("Over the past four hours, " + " ".join(
                    p[0].lower() + p[1:] for p in parts).replace("the user ", "they ")
                    + " [SYNTHETIC fake narrative — not derived from real data]")
                fn.write(json.dumps({"uid": uid, "ema_timestamp": ema_ts,
                                     "rule_based_narrative": rule,
                                     "enhanced_narrative": enhanced}, ensure_ascii=False) + "\n")
                # ----- item-level QA: sample 2–4 items -----
                qrng = random.Random(hash((uid, ts, "qa")) & 0xFFFFFFFF)
                pool = [q for q in EMA_QUESTIONS if row.get(q) not in (None, "")]
                for q in qrng.sample(pool, k=min(len(pool), qrng.randint(2, 4))):
                    sc = float(row[q])
                    ans = (f"The user {_freq(sc)} {_ITEM_TEXT[q]}. "
                           f"[SYNTHETIC fake answer — not real data]")
                    fq.write(json.dumps({"uid": uid, "ema_timestamp": ema_ts,
                                         "question_key": f"ema_{q}",
                                         "enhanced_answer": ans}, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# dataset_info.json (LLaMA-Factory dataset registry; mirrors ChatTS-Training)
# --------------------------------------------------------------------------- #
def gen_dataset_info(out: Path) -> None:
    cols = {"prompt": "input", "response": "output", "timeseries": "timeseries"}
    info = {
        "narrative_dataset": {"file_name": "narrative_dataset/train.jsonl", "columns": cols},
        "qa_dataset": {"file_name": "qa_dataset/train.jsonl", "columns": cols},
        "align_random": {"file_name": "align_random/train.jsonl", "columns": cols},
        "ift": {"file_name": "ift/train.jsonl", "columns": cols},
        # `align_256` in the real setup comes from the ChatTS public release; the
        # fake repo reuses align_random as a stand-in for the alignment channel.
        "align_256": {"file_name": "align_random/train.jsonl", "columns": cols},
    }
    with open(out / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


# --------------------------------------------------------------------------- #
# Final training jsonl (the <ts><ts/> format that stage-1/stage-2 consume)
# --------------------------------------------------------------------------- #
PLACEHOLDER = "<ts><ts/>"
METRICS = [
    ("Heart rate", "beats per minute, bpm", 1440),
    ("Pseudoactigraphy", "Garmin watch accelerometer, product of ZCR count and energy", 480),
    ("Steps per minute", "", 240),
    ("Stress level", "Garmin watch estimate related to HRV", 240),
    ("GPS longitude", "coordinates", 24),
    ("GPS latitude", "coordinates", 24),
    ("Phone unlock status", "binary events per minute", 240),
]


def _metrics_block(rng: random.Random) -> str:
    style = rng.randint(0, 2)
    if style == 0:
        parts = [f"{n} ({d}) is of length {l}: {PLACEHOLDER}" if d else f"{n} is of length {l}: {PLACEHOLDER}"
                 for n, d, l in METRICS]
        return "Your task is to analyze the user's state based on these seven inputs: " + "; ".join(parts) + ";"
    if style == 1:
        return "Metrics table (name — length — placeholder):\n" + "\n".join(
            f"{n} — {l} — {PLACEHOLDER}" for n, _, l in METRICS)
    return "Seven sensor streams:\n" + "\n".join(
        f"- {n} — length {l}: {PLACEHOLDER}" for n, _, l in METRICS)


def _fake_series(length: int, kind: int, rng: np.random.Generator) -> List[float]:
    x = np.arange(length)
    if kind == 4 or kind == 5:  # gps lon/lat
        return [round(float(v), 7) for v in (GPS_CENTER_LON if kind == 4 else GPS_CENTER_LAT)
                + rng.normal(0, 0.002, length)]
    if kind == 6:  # unlock 0/1
        return [float(int(v)) for v in (rng.random(length) < 0.04).astype(int)]
    base = 50 + 20 * np.sin(2 * np.pi * x / max(8, length / 6.0))
    return [round(float(v), 3) for v in np.maximum(0.0, base + rng.normal(0, 4, length))]


def _split_for(uid: str) -> str:
    # deterministic participant-level split, ~70/15/15 across the 4 fake users
    order = sorted(_uid(i) for i in range(1, N_USERS + 1))
    idx = order.index(uid)
    n_train = max(1, int(N_USERS * 0.70))
    n_val = max(1, int(N_USERS * 0.15))
    if idx < n_train:
        return "train"
    if idx < n_train + n_val:
        return "validation"
    return "test"


def gen_training_jsonl(out: Path, ema_times: Dict[str, List[datetime]]) -> None:
    # Build directly from the enriched jsonl + a synthetic timeseries per (uid, ts).
    nar = [json.loads(l) for l in open(out / "enriched_narratives.jsonl", encoding="utf-8")]
    qa = [json.loads(l) for l in open(out / "enriched_qas.jsonl", encoding="utf-8")]
    questions = {q["question_number"]: q["variants"] for q in QUESTION_BANK}

    def ts_for(uid: str, ema_ts: str) -> List[List[float]]:
        rng = np.random.default_rng(hash((uid, ema_ts)) & 0xFFFFFFFF)
        return [_fake_series(L, k, rng) for k, L in enumerate(FINAL_TS_LENGTHS)]

    def qnum(qkey: str):
        d = qkey.replace("ema_Q", "")
        if d.startswith("11"):
            return "11(0)"  # fold all Q11 sub-items onto a single bank entry for simplicity
        try:
            return int(d.split("(")[0])
        except ValueError:
            return None

    nar_rows, qa_rows = [], []
    rrng = random.Random(99)
    for rec in nar:
        uid, ema_ts = rec["uid"], rec["ema_timestamp"]
        inp = _metrics_block(rrng) + (
            f"\n\nsleep duration is {rrng.uniform(5.0, 9.0):.3f} hours\n"
            f"conversation length is {rrng.randint(0, 3000)} seconds\n\nQuestions:\n- "
            + rrng.choice(questions["overall"]))
        nar_rows.append({"uid": uid, "input": inp, "timeseries": ts_for(uid, ema_ts),
                         "output": rec["enhanced_narrative"], "split": _split_for(uid)})
    for rec in qa:
        uid, ema_ts = rec["uid"], rec["ema_timestamp"]
        qn = qnum(rec["question_key"])
        qtext = rrng.choice(questions.get(qn, questions[1])) if qn is not None else rrng.choice(questions[1])
        inp = _metrics_block(rrng) + (
            f"\n\nsleep duration is {rrng.uniform(5.0, 9.0):.3f} hours\n"
            f"conversation length is {rrng.randint(0, 3000)} seconds\n\nQuestions:\n- " + qtext)
        qa_rows.append({"uid": uid, "input": inp, "timeseries": ts_for(uid, ema_ts),
                        "output": rec["enhanced_answer"], "split": _split_for(uid)})

    def dump_splits(rows, name):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        for split in ("train", "validation", "test"):
            sel = [{k: v for k, v in r.items() if k != "split"} for r in rows if r["split"] == split]
            fn = {"train": "train", "validation": "validation", "test": "test"}[split]
            with open(d / f"{fn}.jsonl", "w", encoding="utf-8") as f:
                for r in sel:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump_splits(nar_rows, "narrative_dataset")
    dump_splits(qa_rows, "qa_dataset")

    # ---- align_random / ift: small generic time-series instruction data ----
    def generic_rows(n: int, length: int, instr: str):
        rows = []
        for i in range(n):
            grng = np.random.default_rng(1000 + i)
            rows.append({"input": f"This time series is of length {length}: {PLACEHOLDER}, {instr}",
                         "output": f"[SYNTHETIC fake reference output #{i}]",
                         "timeseries": [_fake_series(length, 0, grng)]})
        return rows

    for name, rows in [("align_random", generic_rows(40, 256, "describe its trend and any periodicity.")),
                       ("ift", generic_rows(20, 256, "answer: what is the overall trend?"))]:
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "train.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Examples + SEED file
# --------------------------------------------------------------------------- #
def gen_examples_and_seed(out: Path, seed: int) -> None:
    (out / "SEED").write_text(f"{seed}\n", encoding="utf-8")
    repo_root = out.parent.parent
    ex = repo_root / "examples"
    ex.mkdir(parents=True, exist_ok=True)
    sample = {}
    for name in ("narrative_dataset", "qa_dataset", "align_random", "ift"):
        p = out / name / "train.jsonl"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                first = f.readline().strip()
            if first:
                sample[name] = json.loads(first)
    with open(ex / "one_sample_each.jsonl", "w", encoding="utf-8") as f:
        for k, v in sample.items():
            # truncate the long timeseries for readability in the example file
            v2 = dict(v)
            if "timeseries" in v2:
                v2["timeseries"] = [s[:8] + ["..."] for s in v2["timeseries"]]
            f.write(json.dumps({"_dataset": k, **v2}, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate LENS synthetic fake data.")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent,
                    help="output directory (default: this data/fake/ dir)")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--all", action="store_true", help="generate everything (default if no flag given)")
    ap.add_argument("--raw-only", action="store_true", help="only the raw sensor CSVs + ema.csv")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    ema_times = _ema_times()
    print(f"[fake] {N_USERS} users x {N_DAYS} days x {len(EMA_HOURS)} EMA/day "
          f"= {sum(len(v) for v in ema_times.values())} EMAs; seed={args.seed}")

    print("[fake] raw sensors ...")
    gen_raw_sensors(out, ema_times)
    print("[fake] ema.csv + filtered_uids.json ...")
    gen_ema(out, ema_times)
    if args.raw_only:
        print("[fake] done (raw-only).")
        return
    print("[fake] Questions.json ...")
    gen_questions(out)
    print("[fake] enriched_narratives.jsonl / enriched_qas.jsonl ...")
    gen_enriched(out, ema_times)
    print("[fake] dataset_info.json ...")
    gen_dataset_info(out)
    print("[fake] training jsonl (narrative / qa / align_random / ift) ...")
    gen_training_jsonl(out, ema_times)
    print("[fake] examples/one_sample_each.jsonl + SEED ...")
    gen_examples_and_seed(out, args.seed)
    print("[fake] done. Output under:", out)


if __name__ == "__main__":
    main()
