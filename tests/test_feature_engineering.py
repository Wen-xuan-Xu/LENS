"""Smoke test for lens.feature_engineering on the synthetic fake data.

Runs the two CLI entry points end-to-end via subprocess and checks that
feature_rows.csv / filtered_feature_rows.csv exist with the expected columns
and a positive number of rows.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_ROOT = REPO_ROOT / "data" / "fake"

FEATURE_ROW_COLUMNS = [
    "uid", "response_time",
    "ema_Q1", "ema_Q2", "ema_Q3", "ema_Q4", "ema_Q5", "ema_Q6", "ema_Q7",
    "ema_Q8", "ema_Q9", "ema_Q10", "ema_Q11(0)", "ema_Q11(1)", "ema_Q11(2)",
    "ema_Q12", "ema_Q13", "ema_Q14",
    "hr_window", "hr_count",
    "zcr_first", "zcr_last", "zcr_count",
    "steps_first", "steps_count",
    "stress_window", "stress_count",
    "gps_lat", "gps_lon", "gps_count",
    "convo_duration", "sleep_duration",
]
FILTERED_EXTRA_COLUMNS = ["unlock_min", "unlock_min_count",
                          "rule_based_narrative", "enhanced_narrative"]


def _run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=REPO_ROOT, check=True)


@pytest.fixture(scope="module")
def fake_data() -> Path:
    if not (FAKE_ROOT / "raw_sensors").is_dir():
        _run(str(FAKE_ROOT / "generate.py"), "--out", str(FAKE_ROOT), "--all")
    assert (FAKE_ROOT / "ema.csv").exists()
    return FAKE_ROOT


def test_feature_engineering_cli(tmp_path, fake_data):
    feature_rows = tmp_path / "feature_rows.csv"
    filtered = tmp_path / "filtered_feature_rows.csv"

    _run("-m", "lens.feature_engineering.build_feature_rows",
         "--data-root", str(fake_data), "--output", str(feature_rows))
    assert feature_rows.exists()
    df = pd.read_csv(feature_rows)
    assert len(df) > 0
    for col in FEATURE_ROW_COLUMNS:
        assert col in df.columns, f"missing column {col} in feature_rows.csv"

    _run("-m", "lens.feature_engineering.add_unlock_and_narratives",
         "--data-root", str(fake_data), "--in", str(feature_rows), "--out", str(filtered))
    assert filtered.exists()
    df2 = pd.read_csv(filtered)
    assert len(df2) > 0
    for col in FEATURE_ROW_COLUMNS + FILTERED_EXTRA_COLUMNS:
        assert col in df2.columns, f"missing column {col} in filtered_feature_rows.csv"
