#!/usr/bin/env python3
"""Build the HuggingFace Arrow ``DatasetDict`` for LENS narrative + QA training.

This is step (1) of the verified data-build chain (CLAUDE.md §3.2):

    filtered_feature_rows.csv  +  Questions.json
      + enriched_narratives.jsonl  + enriched_qas.jsonl
            │  build_dataset.py   (this file)
            ▼
    HF DatasetDict (Arrow), columns = {uid, input, timeseries, output},
    splits = train/validation/test (70/15/15 by participant),
    time-series placeholder written as  <ts></ts>
            │  convert_hf_to_jsonl.py   →  jsonl with <ts></ts>
            │  fix_ts_tokens.py         →  jsonl with <ts><ts/>   (final)
            ▼
    what stage-1 / stage-2 SFT consumes

For each EMA, eight raw time-series columns from the feature rows
(``hr_window``, ``zcr_first``, ``zcr_last``, ``steps_first``, ``stress_window``,
``gps_lon``, ``gps_lat``, ``unlock_min``) are parsed and cleaned, the
pseudoactigraphy stream is computed as the element-wise product of
``zcr_first`` and ``zcr_last``, and the final per-sample ``timeseries`` is the
ordered list ``[hr_window, zcr_prod, steps_first, stress_window, gps_lon,
gps_lat, unlock]`` (7 streams matching the 7 ``<ts></ts>`` placeholders in the
prompt).

If the configured feature-rows CSV is absent (e.g. the feature-engineering
subpackage has not been run yet), a minimal stand-in is synthesized from
``ema.csv`` + the enriched JSONLs so the rest of the chain still runs.
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm

# --------------------------------------------------------------------------- #
# Column definitions
# --------------------------------------------------------------------------- #
TS_COLUMNS_RAW = [
    "hr_window",
    "zcr_first",
    "zcr_last",
    "steps_first",
    "stress_window",
    "gps_lon",
    "gps_lat",
    "unlock_min",
]

FINAL_TS_ORDER = [
    "hr_window",
    "zcr_prod",
    "steps_first",
    "stress_window",
    "gps_lon",
    "gps_lat",
    "unlock",
]

# (display name, description, expected length, placeholder) -- the placeholder
# is the *pre-fix* token; fix_ts_tokens.py later rewrites <ts></ts> -> <ts><ts/>.
METRICS_ORDERED: List[Tuple[str, str, int, str]] = [
    ("Heart rate", "beats per minute, bpm", 1440, "<ts></ts>"),
    ("Pseudoactigraphy", "Garmin watch accelerometer, product of ZCR count and energy", 480, "<ts></ts>"),
    ("Steps per minute", "", 240, "<ts></ts>"),
    ("Stress level", "Garmin watch estimate related to HRV", 240, "<ts></ts>"),
    ("GPS longitude", "coordinates", 24, "<ts></ts>"),
    ("GPS latitude", "coordinates", 24, "<ts></ts>"),
    ("Phone unlock status", "binary events per minute", 240, "<ts></ts>"),
]
FINAL_TS_LENGTHS = [m[2] for m in METRICS_ORDERED]

DEFAULT_RANDOM_SEED = 20250903


# --------------------------------------------------------------------------- #
# Time-series parsing / cleaning utilities
# --------------------------------------------------------------------------- #
def clean_list(vals: List[Any], max_abs: float = 1e6) -> List[float]:
    """Replace non-finite / out-of-range entries with the column median (0 if empty)."""
    clean: List[Optional[float]] = []
    for v in vals:
        try:
            f = float(v)
            clean.append(None if (np.isnan(f) or np.isinf(f) or abs(f) > max_abs) else f)
        except Exception:  # noqa: BLE001
            clean.append(None)
    valid = [x for x in clean if x is not None]
    median = float(np.median(valid)) if valid else 0.0
    return [x if x is not None else median for x in clean]


def parse_list_str(x: Any) -> List[float]:
    """Parse a stringified list (``"[1, 2, 3]"``) into a list of numbers."""
    if isinstance(x, list):
        return x
    if pd.isna(x) or x == "":
        return []
    try:
        v = ast.literal_eval(x)
        if isinstance(v, (list, tuple, np.ndarray)):
            return list(v)
    except Exception:  # noqa: BLE001
        pass
    return []


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x) or x == "":
            return default
        return float(x)
    except Exception:  # noqa: BLE001
        return default


def elemwise_product(a: List[float], b: List[float]) -> List[float]:
    """Element-wise product (truncated to the shorter input)."""
    n = min(len(a), len(b))
    return [float(a[i]) * float(b[i]) for i in range(n)] if n else []


# --------------------------------------------------------------------------- #
# Prompt-template variants (the input text wrapping the 7 placeholders)
# --------------------------------------------------------------------------- #
def _template_v1_prose(metrics):
    parts = [f"{n} ({d}) is of length {l}: {p}" if d else f"{n} is of length {l}: {p}"
             for n, d, l, p in metrics]
    return "The dataset contains the following seven metrics for review: " + "; ".join(parts) + ";"


def _template_v2_list(metrics):
    items = [f"- {n} ({d}) — length {l}: {p}" if d else f"- {n} — length {l}: {p}"
             for n, d, l, p in metrics]
    return "Seven metrics were collected:\n" + "\n".join(items)


def _template_v3_key_value(metrics):
    return "Available metrics: " + ", ".join(f"{n} [{l}] {p}" for n, _, l, p in metrics) + "."


def _template_v4_task_instruction(metrics):
    items = [f"{n} (length {l}) {p}" for n, _, l, p in metrics]
    return ("Your task is to analyze the user's state based on these seven inputs: "
            + "; ".join(items) + ";")


def _template_v5_narrative(metrics):
    sentences = [(f"{n}, measured as {d}, has a length of {l} and is represented as {p}." if d
                  else f"{n} has a length of {l} and is represented as {p}.")
                 for n, d, l, p in metrics]
    return "The monitoring system provides seven continuous data streams. " + " ".join(sentences)


def _template_v6_bulleted_verbose(metrics):
    items = [f"* {n}: measured as {d if d else 'raw counts'}, length {l}, data {p}"
             for n, d, l, p in metrics]
    return "The following metrics have been recorded:\n" + "\n".join(items)


def _template_v7_minimal_sentence(metrics):
    return "Collected streams include: " + ", ".join(f"{n} {p}" for n, _, _, p in metrics) + "."


def _template_v8_tabular_text(metrics):
    return "Metrics table (name — length — placeholder):\n" + "\n".join(
        f"{n} — {l} — {p}" for n, _, l, p in metrics)


def _template_v9_research_style(metrics):
    parts = [f"{n} ({d if d else 'no desc'}, {l} samples) {p}" for n, d, l, p in metrics]
    return "For this study, seven streams are available: " + "; ".join(parts) + "."


def _template_v10_summary_intro(metrics):
    parts = [f"{n} ({l}) {p}" for n, _, l, p in metrics]
    return ("To conduct a comprehensive assessment, seven distinct sensor-derived metrics were "
            "considered. " + "; ".join(parts) + ";")


_TEMPLATES = [_template_v1_prose, _template_v2_list, _template_v3_key_value,
              _template_v4_task_instruction, _template_v5_narrative,
              _template_v6_bulleted_verbose, _template_v7_minimal_sentence,
              _template_v8_tabular_text, _template_v9_research_style,
              _template_v10_summary_intro]


def build_metrics_block(rng: random.Random) -> str:
    """Render the 7-metric block (fixed order, randomly varied phrasing)."""
    return rng.choice(_TEMPLATES)(METRICS_ORDERED)


# --------------------------------------------------------------------------- #
# Question bank + enriched-label loaders
# --------------------------------------------------------------------------- #
def load_questions(json_path: str) -> Tuple[List[str], Dict[int, List[str]]]:
    """Return ``(overall_variants, {question_number: [variants]})``."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    overall_variants: List[str] = []
    by_num: Dict[int, List[str]] = {}
    for item in data:
        num = item.get("question_number")
        variants = item.get("variants") or []
        if not variants:
            continue
        if isinstance(num, str) and num.strip().lower() == "overall":
            overall_variants.extend(variants)
            continue
        if isinstance(num, str) and num.startswith("11("):
            n = 11
        else:
            try:
                n = int(num)
            except Exception:  # noqa: BLE001
                continue
        by_num.setdefault(n, []).extend(variants)
    return overall_variants, by_num


def load_jsonl_index(jsonl_path: str, time_key: str = "ema_timestamp") -> Dict[Tuple[str, str], dict]:
    """Map a JSONL into ``{(uid, timestamp): record}``."""
    index: Dict[Tuple[str, str], dict] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            uid, ts = str(obj.get("uid")), str(obj.get(time_key))
            if uid and ts:
                index[(uid, ts)] = obj
    return index


def load_qa_triple_and_pair(jsonl_path: str, time_key: str = "ema_timestamp"):
    """Return ``(qa_triple[(uid, ts, qkey)], qa_pair[(uid, ts)] -> [records])``."""
    qa_triple: Dict[tuple, dict] = {}
    qa_pair: Dict[tuple, list] = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            uid, ts, qk = str(obj.get("uid")), str(obj.get(time_key)), str(obj.get("question_key"))
            if not (uid and ts and qk):
                continue
            qa_triple[(uid, ts, qk)] = obj
            qa_pair[(uid, ts)].append(obj)
    return qa_triple, qa_pair


_QKEY_RE = re.compile(r"ema_Q(\d+)(?:\(\d+\))?$")


def extract_qnum_from_key(qkey: str) -> Optional[int]:
    if not isinstance(qkey, str):
        return None
    m = _QKEY_RE.search(qkey.strip())
    if not m:
        return None
    n = int(m.group(1))
    return 11 if n == 11 else n


# --------------------------------------------------------------------------- #
# Input-text builders
# --------------------------------------------------------------------------- #
def build_input_text_qa(row: pd.Series, by_num: Dict[int, List[str]], qkey: str,
                        rng: random.Random) -> str:
    metrics = build_metrics_block(rng)
    placeholders = (f"\nsleep duration is {safe_float(row.get('sleep_duration')):.3f} hours\n"
                    f"conversation length is {int(safe_float(row.get('convo_duration')))} seconds\n")
    qnum = extract_qnum_from_key(qkey)
    chosen = [rng.choice(by_num[qnum])] if (qnum is not None and by_num.get(qnum)) else []
    q_block = ("Questions:\n" + "\n".join(f"- {q}" for q in chosen)) if chosen else "Questions:\n- "
    return metrics + "\n\n" + placeholders + "\n" + q_block


def build_input_text_overall(row: pd.Series, overall_vars: List[str], rng: random.Random) -> str:
    metrics = build_metrics_block(rng)
    placeholders = (f"\nsleep duration is {safe_float(row.get('sleep_duration')):.3f} hours\n"
                    f"conversation length is {int(safe_float(row.get('convo_duration')))} seconds\n")
    chosen = [rng.choice(overall_vars)] if overall_vars else []
    q_block = ("Questions:\n" + "\n".join(f"- {q}" for q in chosen)) if chosen else "Questions:\n- "
    return metrics + "\n\n" + placeholders + "\n" + q_block


# --------------------------------------------------------------------------- #
# Split + save
# --------------------------------------------------------------------------- #
def split_by_uid_and_save(build_df: pd.DataFrame, save_dir: Path, preview_name: str,
                          seed: int) -> None:
    """Participant-level 70/15/15 split, saved as an Arrow ``DatasetDict``."""
    save_dir.mkdir(parents=True, exist_ok=True)
    uids = build_df["uid"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(uids)
    n_total = len(uids)
    # Ensure every split is non-empty when there are at least 3 participants.
    if n_total >= 3:
        n_train = max(1, int(round(n_total * 0.70)))
        n_val = max(1, int(round(n_total * 0.15)))
        n_train = min(n_train, n_total - 2)
        n_val = min(n_val, n_total - n_train - 1)
    else:
        n_train, n_val = max(1, n_total - 1), 0

    split_map = {}
    for u in uids[:n_train]:
        split_map[u] = "train"
    for u in uids[n_train:n_train + n_val]:
        split_map[u] = "validation"
    for u in uids[n_train + n_val:]:
        split_map[u] = "test"

    build_df = build_df.copy()
    split_col = build_df["uid"].map(split_map).to_numpy()
    # Keep the data columns only (uid, input, timeseries, output) -- the split is
    # encoded by the DatasetDict keys, matching data/fake/generate.py output.
    ds_all = Dataset.from_pandas(build_df, preserve_index=False)
    dsd = DatasetDict({
        "train": ds_all.select(np.where(split_col == "train")[0].tolist()),
        "validation": ds_all.select(np.where(split_col == "validation")[0].tolist()),
        "test": ds_all.select(np.where(split_col == "test")[0].tolist()),
    })
    dsd.save_to_disk(str(save_dir))
    build_df.head(200).to_json(save_dir / preview_name, orient="records", lines=True,
                               force_ascii=False)
    print(f"[Saved] {save_dir}: "
          f"train={len(dsd['train'])} val={len(dsd['validation'])} test={len(dsd['test'])}")


# --------------------------------------------------------------------------- #
# Fallback: synthesize a minimal feature-rows CSV if one is not available
# --------------------------------------------------------------------------- #
def _synthesize_feature_rows(ema_csv: str, nar_jsonl: str, qa_jsonl: str,
                             seed: int) -> pd.DataFrame:
    """Build a minimal stand-in for ``filtered_feature_rows.csv``.

    Produces one row per (uid, ema_timestamp) that appears in the enriched
    JSONLs, with the ``ema_Q*`` columns from ``ema.csv`` and synthetic raw
    time-series columns of the lengths the build expects.
    """
    rng_np = np.random.default_rng(seed)
    ema = pd.read_csv(ema_csv)
    # Map (uid, "YYYY-mm-dd HH:MM:SS") -> row of ema answers.
    ema_by_key: Dict[Tuple[str, str], dict] = {}
    for _, r in ema.iterrows():
        day = str(r.get("day", ""))
        # ema.csv ``day`` may carry microseconds; the enriched JSONLs do not.
        norm_ts = day.split(".")[0]
        ema_by_key[(str(r.get("uid")), norm_ts)] = r.to_dict()

    keys: List[Tuple[str, str]] = []
    for path in (nar_jsonl, qa_jsonl):
        if not Path(path).exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                k = (str(obj.get("uid")), str(obj.get("ema_timestamp")))
                if k not in keys:
                    keys.append(k)

    def _series(length: int, idx: int, kind: str) -> str:
        if kind == "gps_lon":
            arr = -72.2887 + rng_np.normal(0, 0.002, length)
            return json.dumps([round(float(v), 7) for v in arr])
        if kind == "gps_lat":
            arr = 43.7044 + rng_np.normal(0, 0.002, length)
            return json.dumps([round(float(v), 7) for v in arr])
        if kind == "unlock":
            return json.dumps([float(int(v)) for v in (rng_np.random(length) < 0.04).astype(int)])
        x = np.arange(length)
        base = 50 + 20 * np.sin(2 * np.pi * x / max(8, length / 6.0))
        return json.dumps([round(float(v), 3) for v in np.maximum(0.0, base + rng_np.normal(0, 4, length))])

    rows = []
    ema_qcols = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10",
                 "Q11(0)", "Q11(1)", "Q11(2)", "Q12", "Q13", "Q14"]
    for i, (uid, ts) in enumerate(keys):
        ema_row = ema_by_key.get((uid, ts), {})
        row: Dict[str, Any] = {"uid": uid, "response_time": ts}
        for q in ema_qcols:
            row[f"ema_{q}"] = ema_row.get(q, np.nan)
        row["hr_window"] = _series(1440, i, "hr")
        row["zcr_first"] = _series(480, i, "zcr")  # zcr_count
        row["zcr_last"] = _series(480, i, "zcr")   # energy  -> product = pseudoactigraphy
        row["steps_first"] = _series(240, i, "steps")
        row["stress_window"] = _series(240, i, "stress")
        row["gps_lon"] = _series(24, i, "gps_lon")
        row["gps_lat"] = _series(24, i, "gps_lat")
        row["unlock_min"] = _series(240, i, "unlock")
        row["convo_duration"] = float(rng_np.integers(0, 3000))
        row["sleep_duration"] = float(round(float(rng_np.uniform(5.0, 9.0)), 3))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_paths(cfg: Dict[str, Any], args: argparse.Namespace):
    """Resolve all input/output paths from config + CLI, with sane defaults."""
    here = Path(__file__).resolve().parent
    csv_path = args.csv or cfg.get("csv_path") or cfg.get("feature_rows")
    csv_fallback_root = args.fallback_root or cfg.get("fallback_root")
    questions = args.questions or cfg.get("questions_json") or str(here / "Questions.json")
    nar = args.narratives or cfg.get("narrative_label_jsonl")
    qa = args.qas or cfg.get("qa_label_jsonl")
    save_root = args.save_root or cfg.get("save_root")
    seed = args.seed if args.seed is not None else int(cfg.get("random_seed", DEFAULT_RANDOM_SEED))
    return csv_path, csv_fallback_root, questions, nar, qa, save_root, seed


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Build the LENS narrative + QA Arrow DatasetDict")
    parser.add_argument("--config", help="YAML config file")
    parser.add_argument("--csv", help="Path to filtered_feature_rows.csv")
    parser.add_argument("--fallback-root", help="Dir holding ema.csv + enriched_*.jsonl "
                                                "(used to synthesize feature rows if --csv is missing)")
    parser.add_argument("--questions", help="Path to Questions.json")
    parser.add_argument("--narratives", help="Path to enriched_narratives.jsonl")
    parser.add_argument("--qas", help="Path to enriched_qas.jsonl")
    parser.add_argument("--save-root", help="Output directory root (Arrow datasets go under here)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    csv_path, fallback_root, questions, nar, qa, save_root, seed = _resolve_paths(cfg, args)
    if not save_root:
        raise SystemExit("save_root is required (in config or via --save-root)")
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)

    # Resolve enriched-label / ema paths relative to fallback_root if not given.
    fr = Path(fallback_root) if fallback_root else None
    if not nar and fr:
        nar = str(fr / "enriched_narratives.jsonl")
    if not qa and fr:
        qa = str(fr / "enriched_qas.jsonl")
    if not nar or not qa:
        raise SystemExit("narrative/qa label JSONL paths are required")

    # Load the feature rows, synthesizing a minimal stand-in if necessary.
    if csv_path and Path(csv_path).exists():
        print(f"Loading feature rows from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        if not fr:
            raise SystemExit(f"feature-rows CSV not found ({csv_path!r}) and no --fallback-root given")
        ema_csv = str(fr / "ema.csv")
        print(f"[fallback] {csv_path!r} not found; synthesizing feature rows from {ema_csv}")
        df = _synthesize_feature_rows(ema_csv, nar, qa, seed)

    for c in TS_COLUMNS_RAW:
        if c not in df.columns:
            raise SystemExit(f"feature rows missing required column: {c}")
        df = df[df[c].notna() & (df[c] != "") & (df[c] != "[]")]
    print(f"Rows after prefilter: {len(df)}")

    # Parse + clean the raw TS columns.
    ts_parsed: Dict[str, List[List[float]]] = {}
    for c in tqdm(TS_COLUMNS_RAW, desc="parse-clean columns"):
        parsed = [parse_list_str(x) for x in df[c].tolist()]
        ts_parsed[c] = [clean_list(x, max_abs=1e6) for x in parsed]

    overall_vars, by_num = load_questions(questions)
    _qa_triple, qa_pair = load_qa_triple_and_pair(qa, time_key="ema_timestamp")
    nar_index = load_jsonl_index(nar, time_key="ema_timestamp")

    qa_rows: List[Dict[str, Any]] = []
    nar_rows: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    skipped = 0
    for i, r in tqdm(enumerate(df.itertuples(index=False)), total=len(df), desc="assemble samples"):
        row_raw = {c: ts_parsed[c][i] for c in TS_COLUMNS_RAW}
        final_map = {
            "hr_window": row_raw["hr_window"],
            "zcr_prod": elemwise_product(row_raw["zcr_first"], row_raw["zcr_last"]),
            "steps_first": row_raw["steps_first"],
            "stress_window": row_raw["stress_window"],
            "gps_lon": row_raw["gps_lon"],
            "gps_lat": row_raw["gps_lat"],
            "unlock": row_raw["unlock_min"],
        }
        if any(len(v) == 0 for v in final_map.values()):
            skipped += 1
            continue
        ts_entry = [final_map[k] for k in FINAL_TS_ORDER]
        uid = str(r.uid)
        rtime = str(r.response_time)
        row_series = pd.Series(r._asdict())

        for qa_rec in qa_pair.get((uid, rtime), []):
            qa_rows.append({
                "uid": uid,
                "input": build_input_text_qa(row_series, by_num, qa_rec.get("question_key", ""), rng),
                "timeseries": ts_entry,
                "output": qa_rec.get("enhanced_answer", ""),
            })

        nar_rec = nar_index.get((uid, rtime))
        if nar_rec is not None and overall_vars:
            nar_rows.append({
                "uid": uid,
                "input": build_input_text_overall(row_series, overall_vars, rng),
                "timeseries": ts_entry,
                "output": nar_rec.get("enhanced_narrative", ""),
            })
    print(f"Skipped due to empty final timeseries: {skipped}")

    print("[Saving narrative dataset splits...]")
    if nar_rows:
        nar_df = pd.DataFrame(nar_rows, columns=["uid", "input", "timeseries", "output"])
        split_by_uid_and_save(nar_df, save_root / "narrative_dataset",
                              "preview_narrative_head200.jsonl", seed)
    else:
        print("[Warn] No narrative rows constructed.")

    print("[Saving QA dataset splits...]")
    if qa_rows:
        qa_df = pd.DataFrame(qa_rows, columns=["uid", "input", "timeseries", "output"])
        split_by_uid_and_save(qa_df, save_root / "qa_dataset",
                              "preview_qa_head200.jsonl", seed)
    else:
        print("[Warn] No QA rows constructed.")


if __name__ == "__main__":
    main()
