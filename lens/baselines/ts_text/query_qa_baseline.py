#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS-Text baseline (item-level QA): render the 7 raw series as text into an LLM.

Same idea as ``query_text_baseline.py`` but for the QA dataset: each example
carries one clinical question (extracted from the ``input`` field after
``Questions:``), and the model returns a 1-2 sentence answer. Served via an
OpenAI-compatible endpoint (see ``start_sglang.sh``).

Example
-------
    python -m lens.baselines.ts_text.query_qa_baseline \
        --host 127.0.0.1 --port 30000 --model-name qwen2.5-14B \
        --dataset-jsonl "$DATA_ROOT/qa_dataset/test.jsonl" \
        --output-jsonl out/qa_text_baseline_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

PROMPT_TEMPLATE = (
    "You are a clinical reasoning assistant that interprets physiological and behavioral time-series "
    "data to answer clinical wellbeing questions about the user.\n"
    "You will receive seven time-series streams recorded over the last 4 hours, each represented in "
    "text form, along with two summary variables (sleep duration and conversation length).\n\n"
    "Time-series Inputs\n"
    "1. Heart rate (1 reading per 10 s, length 1440) <ts></ts>\n"
    "2. Pseudoactigraphy (accelerometer movement intensity x zero-crossing rate, length 480) <ts></ts>\n"
    "3. Steps per minute (length 240) <ts></ts>\n"
    "4. Stress level (length 240) <ts></ts>\n"
    "5. GPS longitude (length 24) <ts></ts>\n"
    "6. GPS latitude (length 24) <ts></ts>\n"
    "7. Phone unlock status (binary 0/1 per minute, length 240) <ts></ts>\n\n"
    "{sleep_conversation}\n\n"
    "Question\n{question}\n\n"
    "Answer Requirements\n"
    "- Provide a concise, clinically grounded answer in one or two sentences.\n"
    "- Refer only to the information implied by the time-series data; do not add external facts.\n"
    "- If the data is insufficient, explicitly say so.\nAnswer:\n"
)

_SLEEP_CONV_RE = re.compile(r"sleep duration is .+?\nconversation length is .+? seconds", flags=re.IGNORECASE)


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_question(input_text: str) -> str:
    if not input_text:
        return ""
    parts = input_text.split("Questions:", maxsplit=1)
    if len(parts) < 2:
        return input_text.strip()
    for line in parts[1].strip().splitlines():
        s = line.strip()
        if not s:
            continue
        return s[1:].strip() if s.startswith("-") else s
    return parts[1].strip()


def build_prompt(input_text: str, timeseries) -> Tuple[str, str]:
    question = extract_question(input_text)
    m = _SLEEP_CONV_RE.search(input_text or "")
    parts = PROMPT_TEMPLATE.format(
        sleep_conversation=m.group(0) if m else "", question=question
    ).split("<ts></ts>")
    rebuilt = [parts[0]]
    ts = timeseries or []
    for k in range(len(ts)):
        rebuilt.append(str(ts[k]))
        rebuilt.append(parts[k + 1] if k + 1 < len(parts) else "")
    return "".join(rebuilt), question


def main() -> None:
    p = argparse.ArgumentParser(description="TS-Text baseline (QA) via OpenAI-compatible API.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--base-url", default=None)
    p.add_argument("--model-name", default="qwen2.5-14B")
    p.add_argument("--api-key-env", default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--dataset-jsonl", type=Path, default=Path(os.environ.get("DATA_ROOT", "data/fake")) / "qa_dataset" / "test.jsonl")
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--max-samples", type=int, default=-1)
    args = p.parse_args()

    import openai

    base_url = args.base_url or f"http://{args.host}:{args.port}/v1"
    api_key = os.environ.get(args.api_key_env, "") if args.api_key_env else "EMPTY"
    client = openai.Client(base_url=base_url, api_key=api_key or "EMPTY")

    examples = _read_jsonl(args.dataset_jsonl)
    if args.max_samples >= 0:
        examples = examples[: args.max_samples]

    def work(i: int, example: dict):
        prompt, question = build_prompt(example.get("input", ""), example.get("timeseries"))
        try:
            resp = client.chat.completions.create(
                model=args.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature, max_tokens=args.max_tokens, n=1,
            )
            text = resp.choices[0].message.content
        except Exception as exc:  # noqa: BLE001
            text = ""
            tqdm.write(f"[warn] sample {i} failed: {exc}")
        return i, prompt, question, text

    results: List = [None] * len(examples)
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(work, i, e) for i, e in enumerate(examples)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="ts_text (qa)"):
            i, prompt, question, text = fut.result()
            results[i] = (prompt, question, text)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as fw:
        for i, (prompt, question, text) in enumerate(results):
            fw.write(json.dumps({
                "index": i,
                "uid": examples[i].get("uid"),
                "question": question,
                "prompt": prompt,
                "prediction": text,
                "reference": examples[i].get("output", ""),
            }, ensure_ascii=False) + "\n")
    print(f"Saved to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
