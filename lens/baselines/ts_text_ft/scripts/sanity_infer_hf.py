#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick multi-GPU HF inference sanity check for the TS-Text-FT baseline.

Greedy-decodes a few samples from the narrative/QA ``test.jsonl`` splits with the
training-time ChatML prompt and writes ``<dataset>_sanity.jsonl``. Use it to
confirm the base model + (optional) LoRA adapter load and generate sensibly
before launching a full ``infer_merge_lora.py`` run.

Hardware: GPUs; launch with ``accelerate launch --num_processes N``.

Example
-------
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
      -m lens.baselines.ts_text_ft.scripts.sanity_infer_hf \
      --base-model "$MODEL_DIR" --adapter-path "$OUTPUT_DIR" \
      --data-root "$DATA_ROOT" --output-dir inference_results/ts_text_ft --max-samples 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import List

import torch
from accelerate import Accelerator
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TS-Text-FT HF inference sanity check.")
    p.add_argument("--base-model", type=str, default=os.environ.get("MODEL_DIR", ""))
    p.add_argument("--adapter-path", type=str, default=os.environ.get("OUTPUT_DIR", ""), help="LoRA dir; '' / 'none' to skip.")
    p.add_argument("--data-root", type=str, default=os.environ.get("DATA_ROOT", "data/fake"))
    p.add_argument("--datasets", type=str, nargs="+", default=["narrative_dataset/test.jsonl", "qa_dataset/test.jsonl"])
    p.add_argument("--output-dir", type=str, default="inference_results/ts_text_ft_sanity")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=8, help="<0 = all.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def read_samples(path: Path, max_count: int) -> List[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if 0 <= max_count <= idx:
                break
            items.append({"index": idx, "data": json.loads(line)})
    return items


def build_prompt(sample: dict) -> str:
    instruction = sample["instruction"] if sample.get("instruction") else (sample.get("input", "") or "")
    inp = sample.get("input", "") if sample.get("instruction") else ""
    user_text = f"{instruction}\n{inp}" if inp else instruction
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    )


def main() -> None:
    args = parse_args()
    if not args.base_model:
        raise SystemExit("set --base-model or $MODEL_DIR")
    accelerator = Accelerator()
    torch.backends.cuda.matmul.allow_tf32 = True
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map={"": accelerator.device},
                                                 torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    adapter = args.adapter_path
    if adapter and adapter.lower() != "none":
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter, torch_dtype=torch.bfloat16)
            model.eval()
        except ImportError:
            if accelerator.is_main_process:
                print("[WARN] peft not installed; using the base model only.")

    data_root = Path(args.data_root)
    for rel in args.datasets:
        ds_path = data_root / rel
        if not ds_path.exists():
            accelerator.print(f"[skip] {ds_path} not found")
            continue
        ds_name = Path(rel).parts[0].replace("_dataset", "") or Path(rel).stem
        samples = read_samples(ds_path, args.max_samples)
        if not samples:
            continue
        rank_file = Path(args.output_dir) / f"{ds_name}_sanity_rank{accelerator.process_index}.jsonl"
        with accelerator.split_between_processes(samples) as shard:
            shard = list(shard)
            rank_file.parent.mkdir(parents=True, exist_ok=True)
            with rank_file.open("w", encoding="utf-8") as f:
                if shard:
                    prompts = [build_prompt(it["data"]) for it in shard]
                    refs = [it["data"].get("output", "") for it in shard]
                    n_batches = math.ceil(len(prompts) / args.batch_size)
                    for start in tqdm(range(0, len(prompts), args.batch_size), total=n_batches,
                                      desc=f"{ds_name} p{accelerator.process_index}", leave=False):
                        bp = prompts[start:start + args.batch_size]
                        br = refs[start:start + args.batch_size]
                        bidx = [it["index"] for it in shard[start:start + args.batch_size]]
                        inputs = tokenizer(bp, return_tensors="pt", padding=True).to(model.device)
                        with torch.no_grad():
                            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                        n = inputs["input_ids"].shape[1]
                        for i in range(len(bp)):
                            text = tokenizer.decode(out[i][n:], skip_special_tokens=True).strip()
                            f.write(json.dumps({"dataset": ds_name, "index": int(bidx[i]),
                                                "reference": br[i], "prediction": text}, ensure_ascii=False) + "\n")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            out_file = Path(args.output_dir) / f"{ds_name}_sanity.jsonl"
            with out_file.open("w", encoding="utf-8") as fout:
                for rank in range(accelerator.num_processes):
                    part = Path(args.output_dir) / f"{ds_name}_sanity_rank{rank}.jsonl"
                    if part.exists():
                        fout.write(part.read_text(encoding="utf-8"))
                        part.unlink()
            print(f"[{ds_name}] saved -> {out_file}")


if __name__ == "__main__":
    main()
