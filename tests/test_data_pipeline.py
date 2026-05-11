"""Smoke test for the LENS data-synthesis pipeline.

Runs the four pipeline entry points in order on the synthetic fake data and
asserts the final training JSONL has the expected shape. Runs fully offline
(``--mock-llm``); requires only the core deps + ``datasets``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "fake"
EXPECTED_TS_LENGTHS = [1440, 480, 240, 240, 24, 24, 240]


def _run(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=REPO_ROOT, check=True,
                   capture_output=True, text=True)


@pytest.fixture(scope="module")
def built_pipeline():
    """Generate fake data, then run all four pipeline stages."""
    _run([str(DATA_ROOT / "generate.py"), "--out", str(DATA_ROOT), "--all"])
    _run(["-m", "lens.data_pipeline.run_pipeline",
          "--config", "configs/pipeline/smoke.yaml", "--mock-llm"])
    _run(["-m", "lens.data_pipeline.dataset_build.build_dataset",
          "--config", "configs/pipeline/dataset_build_smoke.yaml"])
    _run(["-m", "lens.data_pipeline.dataset_build.convert_hf_to_jsonl",
          "--root", "data/fake/arrow", "--out", "data/fake"])
    _run(["-m", "lens.data_pipeline.fix_ts_tokens",
          "data/fake/narrative_dataset", "data/fake/qa_dataset"])
    return DATA_ROOT


def test_final_narrative_jsonl_shape(built_pipeline):
    train_path = built_pipeline / "narrative_dataset" / "train.jsonl"
    assert train_path.exists(), f"missing {train_path}"

    lines = [ln for ln in train_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines, "narrative train split is empty"

    for raw in lines:
        obj = json.loads(raw)
        # generate.py emits {uid, input, output, timeseries} per line.
        for key in ("uid", "input", "output", "timeseries"):
            assert key in obj, f"line missing key {key!r}: {sorted(obj)}"
        assert "<ts><ts/>" in obj["input"], "input text should contain the <ts><ts/> placeholder"
        ts = obj["timeseries"]
        assert isinstance(ts, list) and len(ts) == 7, f"expected 7 streams, got {len(ts)}"
        assert [len(s) for s in ts] == EXPECTED_TS_LENGTHS, \
            f"unexpected stream lengths: {[len(s) for s in ts]}"
        for stream in ts:
            assert all(isinstance(x, (int, float)) for x in stream), "non-numeric timeseries value"


def test_other_splits_and_qa_exist(built_pipeline):
    for sub in ("narrative_dataset", "qa_dataset"):
        for split in ("train", "validation", "test"):
            p = built_pipeline / sub / f"{split}.jsonl"
            assert p.exists(), f"missing {p}"
    # Enriched intermediates were (re)written by run_pipeline.
    assert (built_pipeline / "enriched_narratives.jsonl").exists()
    assert (built_pipeline / "enriched_qas.jsonl").exists()
    # The mock-LLM enrichment prefix should be present.
    first = json.loads((built_pipeline / "enriched_narratives.jsonl")
                       .read_text(encoding="utf-8").splitlines()[0])
    assert first["enhanced_narrative"].startswith("[mock-enriched] ")
