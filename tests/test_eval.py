"""Smoke test for the LENS evaluation entry point.

Runs `python -m lens.eval.metrics.compute_nlp_metrics ... --self-test` as a
subprocess (the same command the Makefile / CI uses) and checks:
  * it exits 0 (even when the optional `eval` extra is not installed);
  * it reports ROUGE numbers, and for the trivial pred==ref case the ROUGE-1 F1
    is ~1.0 *if* `rouge-score` is available (otherwise it is correctly skipped).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "one_sample_each.jsonl"


def _run_self_test() -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable, "-m", "lens.eval.metrics.compute_nlp_metrics",
            "--pred", str(EXAMPLE), "--ref", str(EXAMPLE), "--self-test",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_compute_nlp_metrics_self_test_exits_zero():
    proc = _run_self_test()
    assert proc.returncode == 0, f"non-zero exit\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    out = proc.stdout
    assert "[self-test] PASS" in out, out
    # ROUGE numbers must be printed regardless (either ~1.0 or all-zero when skipped).
    assert re.search(r"rouge1\s+P=", out), out

    rouge_skipped = "rouge (needs rouge-score)" in out
    m = re.search(r"rouge1\s+P=[\d.]+\s+R=[\d.]+\s+F1=([\d.]+)", out)
    assert m, f"could not find ROUGE-1 line in:\n{out}"
    r1_f1 = float(m.group(1))
    if rouge_skipped:
        # rouge-score not installed: it must be reported as skipped and F1 is 0.0
        assert r1_f1 == 0.0
        assert "rouge-score not installed" in out or "ROUGE skipped" in out
    else:
        # identical strings -> ROUGE-1 F1 should be essentially 1.0
        assert r1_f1 > 0.99, f"expected ROUGE-1 F1 ~1.0 for pred==ref, got {r1_f1}"


def test_self_test_no_heavy_deps_required(monkeypatch=None):
    # Sanity: importing the module itself must not pull in bert-score / sacrebleu / nltk.
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; import lens.eval.metrics.compute_nlp_metrics as m; "
         "assert 'bert_score' not in sys.modules and 'sacrebleu' not in sys.modules; "
         "print('import-ok')"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "import-ok" in proc.stdout
