#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inference for the TS-Text-FT baseline: (optionally) merge a LoRA adapter, then generate.

Multi-GPU (Accelerate) generation over the narrative/QA ``test.jsonl`` splits
(LLaMA-Factory Alpaca-style records: ``instruction`` / ``input`` / ``output``).
Writes ``<dataset>_results.jsonl`` with ``{dataset, index, reference, prediction}``,
ready for ``lens.eval.metrics.compute_nlp_metrics`` and the LLM-judge pipeline.

Hardware: GPUs required; launch with ``accelerate launch --num_processes N``.

Example
-------
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 \
      -m lens.baselines.ts_text_ft.scripts.infer_merge_lora \
      --base-model "$MODEL_DIR" --adapter-path "$OUTPUT_DIR" \
      --data-root "$DATA_ROOT" --output-dir inference_results/ts_text_ft
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist
from accelerate import Accelerator
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge LoRA (optional) and run inference for TS-Text-FT.")
    p.add_argument("--base-model", type=str, default=os.environ.get("MODEL_DIR", ""), help="Base model (or $MODEL_DIR).")
    p.add_argument("--adapter-path", type=str, default=os.environ.get("OUTPUT_DIR", ""),
                   help="LoRA adapter dir (or $OUTPUT_DIR). Pass '' / 'none' to skip LoRA.")
    p.add_argument("--data-root", type=str, default=os.environ.get("DATA_ROOT", "data/fake"))
    p.add_argument("--datasets", type=str, nargs="+", default=["narrative_dataset/test.jsonl", "qa_dataset/test.jsonl"])
    p.add_argument("--output-dir", type=str, default="inference_results/ts_text_ft")
    p.add_argument("--max-samples", type=int, default=-1, help="<0 = all.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--merge-lora", action="store_true", help="Merge the adapter into the base weights before generating.")
    return p.parse_args()


def read_jsonl(path: Path, max_samples: int) -> List[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            items.append({"index": idx, "data": json.loads(line)})
            if 0 <= max_samples == len(items):
                break
    return items


def build_prompt(sample: dict) -> str:
    instruction = sample.get("instruction", "") or sample.get("input", "") if not sample.get("instruction") else sample["instruction"]
    inp = sample.get("input", "") if sample.get("instruction") else ""
    user_text = f"{instruction}\n{inp}" if inp else instruction
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    )


def gather(accelerator: Accelerator, local: list) -> Optional[list]:
    if accelerator.num_processes == 1:
        return local
    gathered = [None] * accelerator.num_processes
    if hasattr(accelerator, "gather_object"):
        accelerator.gather_object(local, gathered)
    else:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        dist.all_gather_object(gathered, local)
    if accelerator.is_main_process:
        return [r for part in gathered if part for r in part]
    return None


def main() -> None:
    args = parse_args()
    if not args.base_model:
        raise SystemExit("set --base-model or $MODEL_DIR")
    accelerator = Accelerator()
    torch.backends.cuda.matmul.allow_tf32 = True
    device = accelerator.device
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    base_model.eval()

    adapter = args.adapter_path
    if adapter and adapter.lower() != "none":
        from peft import PeftModel

        lora_model = PeftModel.from_pretrained(base_model, adapter, torch_dtype=torch.bfloat16)
        model = lora_model.merge_and_unload() if args.merge_lora else lora_model
    else:
        model = base_model
    model.to(device)
    model.eval()

    def generate_batch(prompts: List[str], tokenizer, model) -> List[str]:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                                 temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, use_cache=False)
        n = inputs["input_ids"].shape[1]
        return [tokenizer.decode(out[i][n:], skip_special_tokens=True).strip() for i in range(len(prompts))]

    data_root = Path(args.data_root)
    for rel in args.datasets:
        ds_path = data_root / rel
        if not ds_path.exists():
            accelerator.print(f"[skip] {ds_path} not found")
            continue
        ds_name = Path(rel).parts[0].replace("_dataset", "") or Path(rel).stem
        entries = read_jsonl(ds_path, args.max_samples)
        if not entries:
            continue
        local_results = []
        with accelerator.split_between_processes(entries) as shard:
            shard = list(shard)
            if shard:
                prompts = [build_prompt(it["data"]) for it in shard]
                refs = [it["data"].get("output", "") for it in shard]
                n_batches = math.ceil(len(prompts) / args.batch_size)
                for start in tqdm(range(0, len(prompts), args.batch_size), total=n_batches,
                                  desc=f"{ds_name} p{accelerator.process_index}", leave=False):
                    bp = prompts[start:start + args.batch_size]
                    br = refs[start:start + args.batch_size]
                    bidx = [it["index"] for it in shard[start:start + args.batch_size]]
                    for idx, ref, pred in zip(bidx, br, generate_batch(bp, tokenizer, model)):
                        local_results.append({"dataset": ds_name, "index": int(idx),
                                               "reference": ref, "prediction": pred})
        merged = gather(accelerator, local_results)
        if merged is not None:
            merged.sort(key=lambda x: x["index"])
            out_file = Path(args.output_dir) / f"{ds_name}_results.jsonl"
            with out_file.open("w", encoding="utf-8") as f:
                for it in merged:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
            accelerator.print(f"[{ds_name}] saved -> {out_file}")
        accelerator.wait_for_everyone()

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    accelerator.print("done.")


if __name__ == "__main__":
    main()
