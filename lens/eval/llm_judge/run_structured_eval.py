#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM-as-judge structured symptom extraction for narratives, via OpenAI Batch API.

Given a single text at a time (a reference summary or a prediction summary), the
judge model returns, for each of the 14 EMA symptom categories, a
``{presence: 0|1, severity: 0..3}`` entry. Run it twice (``--side ref`` and
``--side pred``), merge the two outputs by ``(dataset, index)``, then score with
``lens.eval.metrics.compute_custom_metrics`` (Coverage / Presence Alignment /
Severity Alignment). A ready-made merge helper is in
``lens.eval.scripts/run_llm_judge_pipeline.sh``.

Requires the optional ``eval`` extra (``openai``, ``pydantic``) and
``OPENAI_API_KEY``. Default judge model: ``gpt-4.1-mini``.

Example
-------
    OPENAI_API_KEY=... python -m lens.eval.llm_judge.run_structured_eval \
        --input inference_results/narrative_results.jsonl --input-format jsonl \
        --side pred --output llm_judge/narrative_pred_eval.jsonl --model gpt-4.1-mini
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List


SYSTEM_PROMPT = (
    "You are a strict clinical information extraction and rating engine.\n"
    "You will be given ONE text at a time.\n\n"
    "Your job: for EACH of the 14 EMA symptom categories below, decide:\n"
    "- presence: 0 or 1\n"
    "- severity: 0-3\n\n"
    "CRITICAL RULES:\n"
    "1) Use ONLY the given text. Do NOT infer, assume, or add missing symptoms.\n"
    "2) Do NOT compare with any other text. (You only see one text.)\n"
    "3) If a symptom is NOT explicitly supported by the text, set presence=0 AND severity=0.\n"
    "4) severity must be 0 if presence=0.\n"
    "5) Output MUST be valid JSON matching the provided schema. No extra keys. No explanations.\n\n"
    "Severity scale (ordinal):\n"
    "- 0: absent / not at all\n"
    "- 1: mild / occasionally / somewhat\n"
    "- 2: moderate / often / frequently\n"
    "- 3: severe / almost always / very frequently\n\n"
    "Evaluate the following 14 categories:\n"
    "1. Anhedonia (Interest/Pleasure)\n"
    "2. DepressedMood\n"
    "3. SleepDisturbance\n"
    "4. FatigueEnergy\n"
    "5. AppetiteChange\n"
    "6. SelfWorthGuilt\n"
    "7. Concentration\n"
    "8. PsychomotorChange\n"
    "9. SuicidalIdeation\n"
    "10. SomaticDiscomfort\n"
    "11. AnxietyArousal\n"
    "12. UncontrollableWorry\n"
    "13. NegativeEvent\n"
    "14. OverallSeverity\n"
)

SYMPTOM_CATEGORIES = [
    "Anhedonia", "DepressedMood", "SleepDisturbance", "FatigueEnergy", "AppetiteChange",
    "SelfWorthGuilt", "Concentration", "PsychomotorChange", "SuicidalIdeation",
    "SomaticDiscomfort", "AnxietyArousal", "UncontrollableWorry", "NegativeEvent", "OverallSeverity",
]


def _load_openai_and_pydantic():
    try:
        from openai import OpenAI  # type: ignore
        from pydantic import BaseModel, conint, create_model  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional deps
        raise SystemExit("install the 'eval' extra (`pip install -e .[eval]`) for the LLM judge.") from exc
    return OpenAI, BaseModel, conint, create_model


def get_client(OpenAI):
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAI()


def get_temperature_for_model(model: str) -> float:
    return 1.0 if model.lower().startswith("gpt-5") else 0.0


def build_response_format(BaseModel, conint, create_model) -> Dict[str, object]:
    SymptomPresence = conint(ge=0, le=1)
    SymptomSeverity = conint(ge=0, le=3)

    class SymptomEntry(BaseModel):
        presence: SymptomPresence  # type: ignore[valid-type]
        severity: SymptomSeverity  # type: ignore[valid-type]

    SymptomEvaluation = create_model(
        "SymptomEvaluation", **{cat: (SymptomEntry, ...) for cat in SYMPTOM_CATEGORIES}
    )
    schema = SymptomEvaluation.model_json_schema()
    schema["additionalProperties"] = False
    for d in schema.get("$defs", {}).values():
        d["additionalProperties"] = False
    return {"type": "json_schema", "json_schema": {"name": "SymptomEvaluation", "schema": schema, "strict": True}}


def _load_allowlist(index_file: Path | None) -> set | None:
    if not index_file:
        return None
    if not index_file.exists():
        raise FileNotFoundError(index_file)
    if index_file.suffix.lower() == ".json":
        data = json.loads(index_file.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("--index-file json must be a list")
        return {str(x) for x in data}
    return {ln.strip() for ln in index_file.read_text(encoding="utf-8").splitlines() if ln.strip()}


def row_iter(input_path: Path, input_format: str):
    if input_format == "csv":
        with input_path.open("r", encoding="utf-8") as f:
            yield from csv.DictReader(f)
    elif input_format == "jsonl":
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        raise ValueError(f"unknown input format: {input_format}")


def prepare_batch_requests(input_path, request_path, mapping_path, max_rows, model,
                           reference_column, prediction_column, input_format, side,
                           index_file, response_format) -> Dict[str, Dict[str, str]]:
    allow = _load_allowlist(index_file)
    mapping: Dict[str, Dict[str, str]] = {}
    lines: List[str] = []
    for idx, row in enumerate(row_iter(input_path, input_format)):
        if max_rows is not None and idx >= max_rows:
            break
        reference = row.get(reference_column, "")
        prediction = row.get(prediction_column, "")
        if not reference or not prediction:
            continue
        row_index = row.get("index") or row.get("test_index") or row.get("id")
        row_index = "" if row_index is None else str(row_index)
        if allow is not None and row_index not in allow:
            continue
        if side == "ref":
            user_text, label = reference, "Reference Summary"
        elif side == "pred":
            user_text, label = prediction, "Prediction Summary"
        else:
            raise ValueError(f"unknown side: {side}")
        custom_id = f"sample_{idx}"
        mapping[custom_id] = {"dataset": row.get("dataset") or "", "index": row_index, "side": side}
        body = {
            "model": model,
            "response_format": response_format,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{label}:\n{user_text.strip()}"},
            ],
            "temperature": get_temperature_for_model(model),
        }
        lines.append(json.dumps(
            {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body},
            ensure_ascii=False,
        ))
    if not lines:
        raise RuntimeError("no rows to evaluate.")
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text("\n".join(lines), encoding="utf-8")
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] prepared {len(lines)} batch requests -> {request_path}")
    return mapping


def wait_for_batch(client, batch_id: str, poll_interval: int):
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"[poll] batch {batch_id} status={batch.status}")
        if batch.status in {"completed", "failed", "cancelled", "expired"}:
            return batch
        time.sleep(poll_interval)


def parse_batch_results(raw_output_path: Path, mapping: Dict[str, Dict[str, str]], output_path: Path) -> None:
    evaluations = []
    with raw_output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            cid = entry.get("custom_id")
            response = entry.get("response") or {}
            if response.get("status_code") != 200:
                raise RuntimeError(f"sample {cid} failed: {entry.get('error') or response.get('body')}")
            choices = response["body"].get("choices") or []
            if not choices:
                raise RuntimeError(f"sample {cid} empty output")
            text = (choices[0].get("message") or {}).get("content")
            parsed = json.loads(text)
            meta = mapping.get(cid, {})
            evaluations.append({"dataset": meta.get("dataset"), "index": meta.get("index"),
                                "side": meta.get("side"), "evaluation": parsed})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in evaluations:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[DONE] saved {len(evaluations)} rows to {output_path}")


def merge_ref_pred(ref_path: Path, pred_path: Path, out_path: Path) -> None:
    """Merge ``--side ref`` and ``--side pred`` outputs into one record per sample."""
    def load(p: Path):
        rows = {}
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    o = json.loads(line)
                    rows[(o.get("dataset"), str(o.get("index", "")))] = o
        return rows
    ref_rows, pred_rows = load(ref_path), load(pred_path)
    keys = sorted(set(ref_rows) & set(pred_rows), key=lambda x: (x[0] or "", x[1]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for key in keys:
            re_, pe = ref_rows[key].get("evaluation", {}), pred_rows[key].get("evaluation", {})
            cats = sorted(set(re_) | set(pe))
            eval_out = {
                c: {
                    "ref_presence": int(re_.get(c, {}).get("presence", 0)),
                    "pred_presence": int(pe.get(c, {}).get("presence", 0)),
                    "ref_severity": int(re_.get(c, {}).get("severity", 0)),
                    "pred_severity": int(pe.get(c, {}).get("severity", 0)),
                }
                for c in cats
            }
            f.write(json.dumps({"dataset": ref_rows[key].get("dataset"),
                                "index": ref_rows[key].get("index"), "evaluation": eval_out},
                               ensure_ascii=False) + "\n")
    print(f"[OK] merged {len(keys)} rows -> {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM-as-judge structured symptom evaluation via OpenAI Batch API.")
    p.add_argument("--input", type=Path, default=None, help="Input file (jsonl/csv) with prediction/reference columns.")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--max-rows", type=int, default=None, help="<0 / None = all rows.")
    p.add_argument("--model", type=str, default="gpt-4.1-mini")
    p.add_argument("--reference-column", type=str, default="reference")
    p.add_argument("--prediction-column", type=str, default="prediction")
    p.add_argument("--input-format", choices=["csv", "jsonl"], default="jsonl")
    p.add_argument("--side", choices=["ref", "pred"], default="pred")
    p.add_argument("--index-file", type=Path, default=None, help="Optional allowlist of indices (txt or json list).")
    p.add_argument("--batch-request-file", type=Path, default=None)
    p.add_argument("--batch-mapping-file", type=Path, default=None)
    p.add_argument("--batch-raw-output", type=Path, default=None)
    p.add_argument("--completion-window", type=str, default="24h")
    p.add_argument("--poll-interval", type=int, default=30)
    # convenience: merge two side-outputs without hitting the API
    p.add_argument("--merge", nargs=3, metavar=("REF_JSONL", "PRED_JSONL", "OUT_JSONL"),
                   help="Just merge an existing ref-side and pred-side output, then exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.merge:
        merge_ref_pred(Path(args.merge[0]), Path(args.merge[1]), Path(args.merge[2]))
        return
    if args.input is None or args.output is None:
        raise SystemExit("--input and --output are required (or use --merge).")
    OpenAI, BaseModel, conint, create_model = _load_openai_and_pydantic()
    max_rows = args.max_rows if (args.max_rows is not None and args.max_rows >= 0) else None
    out_dir, stem = args.output.parent, args.output.stem
    req_file = args.batch_request_file or (out_dir / f"{stem}_batch_requests.jsonl")
    map_file = args.batch_mapping_file or (out_dir / f"{stem}_batch_mapping.json")
    raw_file = args.batch_raw_output or (out_dir / f"{stem}_batch_output_raw.jsonl")

    client = get_client(OpenAI)
    response_format = build_response_format(BaseModel, conint, create_model)
    mapping = prepare_batch_requests(args.input, req_file, map_file, max_rows, args.model,
                                     args.reference_column, args.prediction_column, args.input_format,
                                     args.side, args.index_file, response_format)
    with req_file.open("rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(input_file_id=file_obj.id, endpoint="/v1/chat/completions",
                                  completion_window=args.completion_window)
    print(f"[OK] submitted batch {batch.id}")
    batch = wait_for_batch(client, batch.id, args.poll_interval)
    if batch.status != "completed" or not batch.output_file_id:
        raise RuntimeError(f"batch {batch.id} ended with status={batch.status}, errors={getattr(batch, 'errors', None)}")
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.write_bytes(client.files.content(batch.output_file_id).read())
    parse_batch_results(raw_file, mapping, args.output)


if __name__ == "__main__":
    main()
