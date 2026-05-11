#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Token-length distribution for the (text-serialized) training datasets.

Used to pick ``cutoff_len`` for the TS-Text-FT baseline (the series-as-text
prompts are long → it uses 20000, vs ~4000 for the LENS encoder runs). Scans the
``{train,validation,test}.jsonl`` files under each dataset folder, builds the
Qwen/ChatML prompt for each sample, and reports per-split stats (mean, p50/p90/
p95/p99, max, how many exceed ``model_max_length`` and any extra ``--cutoff_lens``).

Requires ``transformers`` (the tokenizer). Multi-process tokenization optional.

Example
-------
    python -m lens.baselines.ts_text_ft.scripts.check_overlength_samples \
        --model "$MODEL_DIR" --data_dir "$DATA_ROOT" \
        --datasets narrative_dataset qa_dataset ift align_random \
        --num_proc 16 --batch_size 1024 --cutoff_lens 4000 8192 20000
"""
from __future__ import annotations

import argparse
import json
from array import array
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from tqdm import tqdm

try:
    from transformers import AutoTokenizer
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("missing dependency: transformers (needed to tokenize). Install transformers.") from exc


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def format_qwen_chatml(user: str, assistant: Optional[str]) -> Tuple[str, str]:
    prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    if assistant is None:
        return prompt, prompt
    return prompt, f"{prompt}{assistant}<|im_end|>\n"


def build_user_text(sample: Dict[str, Any]) -> str:
    inst = sample.get("instruction", "") or sample.get("input", "") or ""
    inp = sample.get("input", "") if sample.get("instruction") else ""
    return f"{inst}\n{inp}" if inp else inst


def percentile(sorted_values: List[int], q: float) -> int:
    if not sorted_values:
        return 0
    if q <= 0:
        return sorted_values[0]
    if q >= 100:
        return sorted_values[-1]
    k = max(1, min(len(sorted_values), int((q / 100.0) * len(sorted_values) + 0.999999999)))
    return sorted_values[k - 1]


def summarize_lengths(lengths: List[int]) -> Dict[str, Any]:
    if not lengths:
        return {"count": 0, "mean": 0.0, "min": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    n = len(lengths)
    sv = sorted(lengths)
    return {"count": n, "mean": sum(lengths) / n, "min": sv[0],
            "p50": percentile(sv, 50), "p90": percentile(sv, 90),
            "p95": percentile(sv, 95), "p99": percentile(sv, 99), "max": sv[-1]}


_WORKER_TOKENIZER = None


def _init_worker_tokenizer(model_dir: str) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)


def _batch_lengths(tok, prompt_texts: List[str], full_texts: List[str]) -> Tuple[List[int], List[int]]:
    pe = tok(prompt_texts, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
    fe = tok(full_texts, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
    return [len(x) for x in pe["input_ids"]], [len(x) for x in fe["input_ids"]]


def _batch_lengths_worker(prompt_texts, full_texts):
    return _batch_lengths(_WORKER_TOKENIZER, prompt_texts, full_texts)


def main() -> None:
    p = argparse.ArgumentParser(description="Token-length distribution for text-serialized datasets.")
    p.add_argument("--model", type=str, required=True, help="Local model dir (or HF id) with tokenizer files.")
    p.add_argument("--data_dir", type=str, required=True, help="Dataset dir (folders with {split}.jsonl).")
    p.add_argument("--datasets", type=str, nargs="+", default=["narrative_dataset", "qa_dataset", "ift", "align_random"])
    p.add_argument("--splits", type=str, nargs="+", default=["train", "validation", "test"])
    p.add_argument("--num_proc", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--max_lines_per_file", type=int, default=0, help="0 = all.")
    p.add_argument("--cutoff_lens", type=int, nargs="*", default=[4000, 8192, 20000])
    p.add_argument("--reports_dir", type=str, default="reports")
    args = p.parse_args()

    model_dir, data_dir, reports_dir = Path(args.model), Path(args.data_dir), Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)
    model_max_len = int(getattr(tokenizer, "model_max_length", 0) or 0)
    if model_max_len <= 0 or model_max_len > 10 ** 9:
        model_max_len = 32768  # tokenizers sometimes report a huge sentinel; pick a sane default for reporting

    executor: Optional[ProcessPoolExecutor] = None
    if args.num_proc and args.num_proc > 1:
        executor = ProcessPoolExecutor(max_workers=args.num_proc, initializer=_init_worker_tokenizer, initargs=(str(model_dir),))
    max_inflight = 2 * args.num_proc if executor else 0

    summary_rows: List[Dict[str, Any]] = []
    for ds in args.datasets:
        ds_dir = data_dir / ds
        if not ds_dir.exists():
            print(f"[WARN] dataset dir not found, skip: {ds_dir}")
            continue
        for split in args.splits:
            fp = ds_dir / f"{split}.jsonl"
            if not fp.exists():
                continue
            prompt_lens, full_lens = array("I"), array("I")
            over_max = 0
            cutoff_over = {c: 0 for c in args.cutoff_lens}
            prompt_buf, full_buf, inflight, total = [], [], [], 0

            def consume(p_lens, f_lens):
                nonlocal over_max
                for pl, fl in zip(p_lens, f_lens):
                    prompt_lens.append(pl)
                    full_lens.append(fl)
                    if pl > model_max_len or fl > model_max_len:
                        over_max += 1
                    for c in args.cutoff_lens:
                        if fl > c:
                            cutoff_over[c] += 1

            def flush():
                nonlocal prompt_buf, full_buf, inflight
                if not prompt_buf:
                    return
                if executor is None:
                    consume(*_batch_lengths(tokenizer, prompt_buf, full_buf))
                else:
                    inflight.append(executor.submit(_batch_lengths_worker, prompt_buf, full_buf))
                    if len(inflight) >= max_inflight:
                        consume(*inflight.pop(0).result())
                prompt_buf, full_buf = [], []

            for _line_no, sample in tqdm(iter_jsonl(fp), desc=f"scan {ds}/{split}", unit="ex"):
                total += 1
                if args.max_lines_per_file and total > args.max_lines_per_file:
                    break
                pt, ft = format_qwen_chatml(build_user_text(sample), sample.get("output", "") or "")
                prompt_buf.append(pt)
                full_buf.append(ft)
                if len(prompt_buf) >= args.batch_size:
                    flush()
            flush()
            for fut in inflight:
                consume(*fut.result())

            row: Dict[str, Any] = {
                "dataset": ds, "split": split, "file": str(fp), "model_max_length": model_max_len,
                "total": total, "over_model_max": over_max,
                "over_model_max_ratio": (over_max / total) if total else 0.0,
                "prompt_tokens": summarize_lengths(list(prompt_lens)),
                "total_tokens": summarize_lengths(list(full_lens)),
            }
            if args.cutoff_lens and total:
                row["over_cutoff"] = {str(c): {"count": cutoff_over[c], "ratio": cutoff_over[c] / total} for c in args.cutoff_lens}
            summary_rows.append(row)

    if executor is not None:
        executor.shutdown(wait=True)

    out = {
        "model": str(model_dir), "data_dir": str(data_dir), "model_max_length": model_max_len,
        "datasets": args.datasets, "splits": args.splits, "cutoff_lens": args.cutoff_lens,
        "by_dataset_split": summary_rows,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = reports_dir / f"length_stats_{ts}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nDone. summary -> {out_path}")


if __name__ == "__main__":
    main()
