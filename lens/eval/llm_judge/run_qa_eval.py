#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM-as-judge (Severity Alignment) for item-level QA, via the OpenAI Batch API.

For each (reference, prediction) pair the judge model rates the severity of the
described symptom/behavior on a 0..3 scale for *each* text, emitting
``{ref_severity, pred_severity}``. Score them with
``lens.eval.metrics.compute_qa_metrics``.

Requires the optional ``eval`` extra (``openai``, ``pydantic``) and the
``OPENAI_API_KEY`` environment variable. The default judge model is
``gpt-4.1-mini`` (the model used in the paper).

Example
-------
    OPENAI_API_KEY=... python -m lens.eval.llm_judge.run_qa_eval \
        --input inference_results/qa_results.jsonl --input-format jsonl \
        --reference-column reference --prediction-column prediction \
        --output llm_judge/qa_eval.jsonl --model gpt-4.1-mini
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
    "You are a clinical evaluation model. Your task is to assess the severity of a symptom or behavior "
    "described in two texts: a ground-truth reference and a model-generated prediction.\n\n"
    "For EACH text, output a severity score from 0 to 3:\n"
    "- 0: No symptom / absent / not at all\n"
    "- 1: Mild / occasionally / somewhat\n"
    "- 2: Moderate / often / frequently\n"
    "- 3: Severe / almost always / very frequently\n\n"
    "You must output exactly two fields:\n"
    "- ref_severity: severity score (0-3) for the reference text\n"
    "- pred_severity: severity score (0-3) for the prediction text\n\n"
    "Base your judgment on the semantic intensity and frequency descriptors in each text. "
    "Do not add explanations or other fields."
)


def _load_openai_and_pydantic():
    try:
        from openai import OpenAI  # type: ignore
        from openai.types import FileObject  # type: ignore
        from pydantic import BaseModel, conint  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional deps
        raise SystemExit("install the 'eval' extra (`pip install -e .[eval]`) for the LLM judge.") from exc
    return OpenAI, FileObject, BaseModel, conint


def get_client(OpenAI):
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAI()


def get_temperature_for_model(model: str) -> float:
    return 1.0 if model.lower().startswith("gpt-5") else 0.0


def build_prompt(reference: str, prediction: str, question: str | None = None) -> str:
    parts = []
    if question:
        parts.append(f"Question: {question.strip()}\n")
    parts.append(f"Reference: {reference.strip()}\n")
    parts.append(f"Prediction: {prediction.strip()}")
    return "\n".join(parts)


def build_response_format(BaseModel, conint) -> Dict[str, object]:
    SeverityScore = conint(ge=0, le=3)

    class SeverityPair(BaseModel):
        ref_severity: SeverityScore  # type: ignore[valid-type]
        pred_severity: SeverityScore  # type: ignore[valid-type]

    schema = SeverityPair.model_json_schema()
    schema["additionalProperties"] = False
    return {"type": "json_schema", "json_schema": {"name": "SeverityPair", "schema": schema, "strict": True}}


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
    elif input_format == "json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        yield from data.get("results", [])
    else:
        raise ValueError(f"unknown input format: {input_format}")


def prepare_batch_requests(input_path, request_path, mapping_path, max_rows, model,
                           reference_column, prediction_column, question_column, input_format,
                           response_format) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    lines: List[str] = []
    for idx, row in enumerate(row_iter(input_path, input_format)):
        if max_rows is not None and idx >= max_rows:
            break
        reference = row.get(reference_column, "")
        prediction = row.get(prediction_column, "")
        if not reference or not prediction:
            continue
        question = row.get(question_column, "") if question_column else ""
        custom_id = f"qa_{idx}"
        mapping[custom_id] = {
            "dataset": row.get("dataset", "qa"),
            "index": str(row.get("index") or row.get("test_index") or idx),
            "question": question,
        }
        body = {
            "model": model,
            "response_format": response_format,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(reference, prediction, question)},
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
            evaluations.append({
                "dataset": meta.get("dataset"),
                "index": meta.get("index"),
                "question": meta.get("question"),
                "ref_severity": parsed.get("ref_severity"),
                "pred_severity": parsed.get("pred_severity"),
            })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in evaluations:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[DONE] saved {len(evaluations)} rows to {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM-as-judge QA severity evaluation via OpenAI Batch API.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-rows", type=int, default=None, help="<0 / None = all rows.")
    p.add_argument("--model", type=str, default="gpt-4.1-mini")
    p.add_argument("--reference-column", type=str, default="reference")
    p.add_argument("--prediction-column", type=str, default="prediction")
    p.add_argument("--question-column", type=str, default="")
    p.add_argument("--input-format", choices=["csv", "jsonl", "json"], default="jsonl")
    p.add_argument("--batch-request-file", type=Path, default=None)
    p.add_argument("--batch-mapping-file", type=Path, default=None)
    p.add_argument("--batch-raw-output", type=Path, default=None)
    p.add_argument("--completion-window", type=str, default="24h")
    p.add_argument("--poll-interval", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OpenAI, _FileObject, BaseModel, conint = _load_openai_and_pydantic()
    max_rows = args.max_rows if (args.max_rows is not None and args.max_rows >= 0) else None
    out_dir, stem = args.output.parent, args.output.stem
    req_file = args.batch_request_file or (out_dir / f"{stem}_batch_requests.jsonl")
    map_file = args.batch_mapping_file or (out_dir / f"{stem}_batch_mapping.json")
    raw_file = args.batch_raw_output or (out_dir / f"{stem}_batch_output_raw.jsonl")

    client = get_client(OpenAI)
    response_format = build_response_format(BaseModel, conint)
    mapping = prepare_batch_requests(args.input, req_file, map_file, max_rows, args.model,
                                     args.reference_column, args.prediction_column,
                                     args.question_column or None, args.input_format, response_format)
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
