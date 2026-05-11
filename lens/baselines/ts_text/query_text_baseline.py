#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS-Text baseline (narratives): render the 7 raw series as text into an LLM.

This baseline serializes each of the 7 time-series streams (heart rate,
pseudoactigraphy, steps, stress, GPS lon, GPS lat, phone unlock) plus the
sleep/conversation summary variables straight into the prompt as text, and asks
a generic LLM (in the paper: Qwen2.5-14B) for a clinical narrative. The model is
served via an OpenAI-compatible endpoint (e.g. SGLang / vLLM — see
``start_sglang.sh``).

No API key is needed for a local server (``api_key`` is a placeholder); for a
hosted endpoint set ``OPENAI_API_KEY`` and pass ``--api-key-env OPENAI_API_KEY``.

Example
-------
    python -m lens.baselines.ts_text.query_text_baseline \
        --host 127.0.0.1 --port 30000 --model-name qwen2.5-14B \
        --dataset-jsonl "$DATA_ROOT/narrative_dataset/test.jsonl" \
        --output-jsonl out/text_baseline_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from tqdm import tqdm

PROMPT_TEMPLATE = (
    "You are a clinical reasoning assistant that interprets physiological and behavioral time-series "
    "data to infer a user's psychological and physical wellbeing.\n"
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
    "Task\n"
    "Using only the provided textual data, produce a short clinical summary (about one concise "
    "paragraph) describing the user's psychological and physical state over the last 4 hours, "
    "covering: interest/pleasure, mood, sleep, energy/fatigue, appetite, self-esteem, concentration, "
    "psychomotor activity, thoughts of self-harm, physical discomfort, anxiety, uncontrollable worry, "
    "and recent negative events.\n"
    "Do not mention raw numbers, arrays, or sensor names. Do not explain your reasoning.\n"
    "End with a brief statement of the likely mood severity (mild / moderate / severe).\n\n"
    "Output Format\n"
    "Return only the narrative summary paragraph; no bullet points, lists, or section headers.\n"
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


def _placeholder_token(text: str) -> str:
    # accept either the dataset placeholder ("<ts><ts/>") or the prompt template's ("<ts></ts>")
    return "<ts><ts/>" if "<ts><ts/>" in text else "<ts></ts>"


def build_prompt(example: dict) -> str:
    input_text = example.get("input", "") or ""
    m = _SLEEP_CONV_RE.search(input_text)
    prompt = PROMPT_TEMPLATE.format(sleep_conversation=m.group(0) if m else "")
    parts = prompt.split("<ts></ts>")
    ts = example.get("timeseries", []) or []
    rebuilt = [parts[0]]
    for k in range(len(ts)):
        rebuilt.append(str(ts[k]))
        rebuilt.append(parts[k + 1] if k + 1 < len(parts) else "")
    return "".join(rebuilt)


def main() -> None:
    p = argparse.ArgumentParser(description="TS-Text baseline (narratives) via OpenAI-compatible API.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--base-url", default=None, help="Override full base URL (else http://host:port/v1).")
    p.add_argument("--model-name", default="qwen2.5-14B")
    p.add_argument("--api-key-env", default=None, help="Env var holding the API key (else a placeholder).")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--dataset-jsonl", type=Path, default=Path(os.environ.get("DATA_ROOT", "data/fake")) / "narrative_dataset" / "test.jsonl")
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-samples", type=int, default=-1, help="<0 = all.")
    args = p.parse_args()

    import openai  # imported here so `--help` works without the dep

    base_url = args.base_url or f"http://{args.host}:{args.port}/v1"
    api_key = os.environ.get(args.api_key_env, "") if args.api_key_env else "EMPTY"
    client = openai.Client(base_url=base_url, api_key=api_key or "EMPTY")

    examples = _read_jsonl(args.dataset_jsonl)
    if args.max_samples >= 0:
        examples = examples[: args.max_samples]

    def work(i: int, example: dict):
        prompt = build_prompt(example)
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
        return i, prompt, text

    results: List = [None] * len(examples)
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(work, i, e) for i, e in enumerate(examples)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="ts_text (narrative)"):
            i, prompt, text = fut.result()
            results[i] = (prompt, text)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as fw:
        for i, (prompt, text) in enumerate(results):
            fw.write(json.dumps({
                "index": i,
                "uid": examples[i].get("uid"),
                "prompt": prompt,
                "prediction": text,
                "reference": examples[i].get("output", ""),
            }, ensure_ascii=False) + "\n")
    print(f"Saved to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
