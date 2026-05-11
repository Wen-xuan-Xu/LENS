#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HuggingFace Transformers inference for LENS (patch-based TS encoder) models.

Loads a trained LENS checkpoint with ``trust_remote_code=True`` (the ChatTS /
Qwen2TS processor + patch encoder live in the model repo) and runs batched,
multi-GPU generation over the narrative/QA ``test.jsonl`` splits, writing
``<dataset>_results.jsonl`` with ``{dataset, index, reference, prediction}``.

Hardware: 1+ GPU; launch with ``accelerate launch --num_processes N``.

Example
-------
    accelerate launch --num_processes 4 -m lens.eval.inference.hf_inference \
        --model-path "$MODEL_DIR" \
        --data-root "$DATA_ROOT" \
        --datasets narrative_dataset/test.jsonl qa_dataset/test.jsonl \
        --output-dir inference_results \
        --max-samples-per-dataset -1
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LENS HF inference (patch-based TS encoder).")
    p.add_argument("--model-path", type=str, default=os.environ.get("MODEL_DIR", ""),
                   help="Path or HF id of the trained LENS checkpoint (or set $MODEL_DIR).")
    p.add_argument("--data-root", type=str, default=os.environ.get("DATA_ROOT", "data/fake"),
                   help="Root holding the dataset folders (or set $DATA_ROOT).")
    p.add_argument("--datasets", type=str, nargs="+",
                   default=["narrative_dataset/test.jsonl", "qa_dataset/test.jsonl"],
                   help="JSONL paths relative to --data-root; each must have {input, timeseries, output}.")
    p.add_argument("--output-dir", type=str, default="inference_results")
    p.add_argument("--max-samples-per-dataset", type=int, default=-1, help="<0 = all samples.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--do-sample", action="store_true", default=True)
    return p.parse_args()


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_samples(path: Path, indices: List[int]) -> Dict[int, dict]:
    targets = set(indices)
    out: Dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in targets:
                out[i] = json.loads(line)
                if len(out) == len(targets):
                    break
    missing = targets - set(out)
    if missing:
        raise ValueError(f"missing indices {sorted(missing)} in {path}")
    return out


def build_prompt(user_prompt: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def prepare_batch_inputs(batch_samples: List[dict], processor, device: torch.device):
    prompts: List[str] = []
    flattened_ts: List[np.ndarray] = []
    for s in batch_samples:
        prompts.append(build_prompt(s["input"]))
        flattened_ts.extend(np.array(ts, dtype=np.float32) for ts in s["timeseries"])
    encoded = processor(text=prompts, timeseries=flattened_ts, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in encoded.items()}


def main() -> None:
    args = parse_args()
    if not args.model_path:
        raise SystemExit("set --model-path or $MODEL_DIR")
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    max_samples = args.max_samples_per_dataset if args.max_samples_per_dataset >= 0 else None
    batch_size = max(1, args.batch_size)

    accelerator = Accelerator()
    torch.backends.cuda.matmul.allow_tf32 = True
    device = accelerator.device
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, tokenizer=tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, device_map={"": device}, torch_dtype=torch.float32,
    )
    model.eval()

    for rel in args.datasets:
        dataset_path = data_root / rel
        key = Path(rel).parts[0].replace("_dataset", "") or Path(rel).stem
        total = count_lines(dataset_path) if dataset_path.exists() else 0
        if total == 0:
            accelerator.print(f"[skip] {dataset_path} not found / empty")
            continue
        indices = list(range(total))
        if max_samples is not None:
            indices = indices[:max_samples]
        accelerator.print(f"\n==== {key} ({dataset_path}) : {len(indices)} samples ====")

        local_results = []
        with accelerator.split_between_processes(indices) as local_indices:
            local_indices = sorted(local_indices)
            if local_indices:
                samples = load_samples(dataset_path, local_indices)
                n_batches = math.ceil(len(local_indices) / batch_size)
                for start in tqdm(range(0, len(local_indices), batch_size), total=n_batches,
                                  desc=f"{key} p{accelerator.process_index}", leave=False):
                    bidx = local_indices[start:start + batch_size]
                    bs = [samples[i] for i in bidx]
                    enc = prepare_batch_inputs(bs, processor, device)
                    with torch.inference_mode():
                        gen = model.generate(
                            **enc, max_new_tokens=args.max_new_tokens, do_sample=args.do_sample,
                            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                        )
                    prefix_len = enc["input_ids"].shape[1]
                    for i, idx in enumerate(bidx):
                        text = tokenizer.decode(gen[i][prefix_len:], skip_special_tokens=True)
                        local_results.append({"dataset": key, "index": idx,
                                               "reference": bs[i]["output"], "prediction": text})

        if accelerator.num_processes > 1:
            gathered = [None] * accelerator.num_processes
            dist.all_gather_object(gathered, local_results)
            if accelerator.is_main_process:
                merged = [r for part in gathered for r in part]
                merged.sort(key=lambda x: x["index"])
                out_file = output_dir / f"{key}_results.jsonl"
                with out_file.open("w", encoding="utf-8") as f:
                    for r in merged:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                accelerator.print(f"saved -> {out_file}")
        else:
            local_results.sort(key=lambda x: x["index"])
            out_file = output_dir / f"{key}_results.jsonl"
            with out_file.open("w", encoding="utf-8") as f:
                for r in local_results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"saved -> {out_file}")
        accelerator.wait_for_everyone()

    del model, processor, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    accelerator.free_memory()
    if accelerator.num_processes > 1:
        try:
            dist.destroy_process_group()
        except RuntimeError:
            pass
    accelerator.print("\ndone.")


if __name__ == "__main__":
    main()
