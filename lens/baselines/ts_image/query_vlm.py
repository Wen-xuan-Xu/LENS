#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS-Image baseline: render the 7 raw series as a chart image into Qwen2.5-VL.

Each test example's seven streams are plotted as a multi-panel chart (rendered
on the fly from the dataset's ``timeseries`` field, or read from a pre-built
``--plots-dir`` of ``*_test_idx{idx}_uid_{uid}.png`` files), then sent to a
Qwen2.5-VL model (7B or 32B) served via the SGLang offline engine, with optional
labeled few-shot example images prepended. Writes a results JSONL (and CSV) with
``{index, uid, prompt, prediction, reference, question}``.

NOTE: the upstream code lived in a directory mis-named ``Qwen3-VL`` but the model
is Qwen2.5-VL — that's what ``ts_image`` targets.

Hardware: GPUs required (SGLang engine, tp/dp configurable).

Example
-------
    python -m lens.baselines.ts_image.query_vlm \
        --model-path "$VLM_MODEL_DIR" --model-size 32b \
        --dataset-jsonl "$DATA_ROOT/narrative_dataset/test.jsonl" \
        --dataset-type narrative --output-jsonl out/vlm_responses_32b.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from .prompt import PROMPT_TEMPLATE_NARRATIVE, PROMPT_TEMPLATE_QA

CHAT_TEMPLATE = "qwen2-vl"
FILENAME_PATTERN = re.compile(r"^(?P<prefix>.+?)_(?P<split>train|validation|test)_idx(?P<index>\d+)_uid_(?P<uid>.+)$")
STREAM_NAMES = ["Heart rate", "Pseudoactigraphy", "Steps/min", "Stress level",
                "GPS longitude", "GPS latitude", "Phone unlock"]


def extract_contextual_features(input_text: str) -> str:
    feats = []
    m = re.search(r"sleep duration is ([\d.]+) hours", input_text or "")
    if m:
        feats.append(f"sleep duration is {m.group(1)} hours")
    m = re.search(r"conversation length is ([\d.]+) seconds", input_text or "")
    if m:
        feats.append(f"conversation length is {m.group(1)} seconds")
    return "\n".join(feats) if feats else "No additional features available."


def extract_question(input_text: str) -> Optional[str]:
    if not input_text or "Questions:" not in input_text:
        return None
    for line in input_text.split("Questions:", 1)[1].splitlines():
        s = line.strip()
        if s:
            return s.lstrip("-").strip()
    return None


def format_prompt(dataset_type: str, contextual_features: str, question: Optional[str]) -> str:
    if dataset_type == "qa":
        return PROMPT_TEMPLATE_QA.format(contextual_features=contextual_features,
                                         question=question or "No question provided.")
    return PROMPT_TEMPLATE_NARRATIVE.format(contextual_features=contextual_features)


def render_chart(timeseries: List[List[float]], out_path: Path) -> None:
    """Render the 7 streams as a stacked multi-panel PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(timeseries)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.6 * n))
    if n == 1:
        axes = [axes]
    for i, (ax, series) in enumerate(zip(axes, timeseries)):
        ax.plot(series, linewidth=0.8)
        ax.set_ylabel(STREAM_NAMES[i] if i < len(STREAM_NAMES) else f"stream {i}", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_image(path: str):
    from PIL import Image

    with Image.open(path) as img:
        return img.copy()


def _get_image_token() -> str:
    from sglang.srt.parser.conversation import chat_templates

    return chat_templates[CHAT_TEMPLATE].image_token


def build_prompt(dataset_type: str, contextual_features: str, question: Optional[str],
                 few_shot: List[Dict]) -> str:
    image_token = _get_image_token()
    main = format_prompt(dataset_type, contextual_features, question)
    if not few_shot:
        return f"{image_token}\n{main}"
    sections = ["You will review several labeled reference cases before responding to the new case."
                if dataset_type != "qa" else
                "Here are some examples of similar clinical assessments to help you understand the task:"]
    for i, ex in enumerate(few_shot, 1):
        if dataset_type == "qa":
            sections.append(f"{image_token}\nExample {i}:\nContextual features: {ex['contextual_features']}\n"
                            f"Question: {ex.get('question')}\nAnswer: {ex['reference'].strip()}\n")
        else:
            sections.append(f"{image_token}\nReference case {i} prompt:\n{ex['prompt'].strip()}\n\n"
                            f"Reference case {i} clinician summary:\n{ex['reference'].strip()}\n")
    sections.append("---")
    lead = ("Now answer the following question based on the new case:\n\n" if dataset_type == "qa"
            else "Now analyze the next case following the same instructions:\n\n")
    sections.append(f"{image_token}\n{lead}{main}")
    return "\n\n".join(sections)


def main() -> None:
    p = argparse.ArgumentParser(description="TS-Image baseline (Qwen2.5-VL on rendered charts) via SGLang.")
    p.add_argument("--model-path", default=os.environ.get("VLM_MODEL_DIR", ""), help="Qwen2.5-VL path/HF-id (or $VLM_MODEL_DIR).")
    p.add_argument("--model-size", choices=["7b", "32b"], default="32b")
    p.add_argument("--dataset-type", choices=["narrative", "qa"], default="narrative")
    p.add_argument("--dataset-jsonl", type=Path, default=None,
                   help="JSONL with {input, timeseries, output}. Default: $DATA_ROOT/<type>_dataset/test.jsonl")
    p.add_argument("--plots-dir", type=Path, default=None,
                   help="Optional pre-rendered chart dir (*_test_idx{idx}_uid_{uid}.png). If unset, charts are rendered on the fly.")
    p.add_argument("--few-shot-jsonl", type=Path, default=None, help="Optional labeled examples (train split) for few-shot.")
    p.add_argument("--few-shot-count", type=int, default=0)
    p.add_argument("--few-shot-seed", type=int, default=42)
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--dp-size", type=int, default=1)
    p.add_argument("--mem-fraction", type=float, default=None, help="Default 0.7 (7b) / 0.8 (32b).")
    p.add_argument("--max-samples", type=int, default=-1)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--chart-cache-dir", type=Path, default=Path("./_ts_image_charts"))
    args = p.parse_args()

    if not args.model_path:
        raise SystemExit("set --model-path or $VLM_MODEL_DIR")
    data_root = Path(os.environ.get("DATA_ROOT", "data/fake"))
    dataset_jsonl = args.dataset_jsonl or (data_root / f"{args.dataset_type}_dataset" / "test.jsonl")
    mem_fraction = args.mem_fraction if args.mem_fraction is not None else (0.8 if args.model_size == "32b" else 0.7)

    examples = _read_jsonl(dataset_jsonl)
    if args.max_samples >= 0:
        examples = examples[: args.max_samples]

    # ---- few-shot examples ----
    few_shot: List[Dict] = []
    if args.few_shot_count > 0 and args.few_shot_jsonl is not None:
        pool = _read_jsonl(args.few_shot_jsonl)
        rng = random.Random(args.few_shot_seed)
        for ex in rng.sample(pool, min(args.few_shot_count, len(pool))):
            cf = extract_contextual_features(ex.get("input", ""))
            q = extract_question(ex.get("input", "")) if args.dataset_type == "qa" else None
            img_path = args.chart_cache_dir / f"fewshot_{ex.get('uid','x')}_{hash(json.dumps(ex.get('timeseries',[])))&0xffff}.png"
            render_chart(ex.get("timeseries", []), img_path)
            few_shot.append({"plot_path": str(img_path), "contextual_features": cf, "question": q,
                             "prompt": format_prompt(args.dataset_type, cf, q), "reference": ex.get("output", "")})

    # ---- resolve a chart image per example ----
    def chart_for(idx: int, ex: dict) -> str:
        if args.plots_dir is not None:
            # match *_test_idx{idx}_uid_*.png
            for f in sorted(args.plots_dir.glob(f"*_test_idx{idx}_uid_*.png")):
                return str(f)
            raise FileNotFoundError(f"no chart for index {idx} in {args.plots_dir}")
        img_path = args.chart_cache_dir / f"{args.dataset_type}_test_idx{idx}_uid_{ex.get('uid','x')}.png"
        if not img_path.exists():
            render_chart(ex.get("timeseries", []), img_path)
        return str(img_path)

    from sglang import Engine

    llm = Engine(model_path=args.model_path, chat_template=CHAT_TEMPLATE,
                 mem_fraction_static=mem_fraction, tp_size=args.tp_size, dp_size=args.dp_size)
    sampling_params = {"temperature": args.temperature, "top_p": args.top_p,
                       "top_k": args.top_k, "max_new_tokens": args.max_new_tokens}
    few_shot_imgs = [_load_image(ex["plot_path"]) for ex in few_shot]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as fw:
        for idx, ex in enumerate(tqdm(examples, desc=f"ts_image ({args.dataset_type})")):
            cf = extract_contextual_features(ex.get("input", ""))
            q = extract_question(ex.get("input", "")) if args.dataset_type == "qa" else None
            prompt = build_prompt(args.dataset_type, cf, q, few_shot)
            images = list(few_shot_imgs)
            try:
                images.append(_load_image(chart_for(idx, ex)))
                resp = llm.generate(prompt=prompt, image_data=images, sampling_params=sampling_params)
                text = resp["text"]
            except Exception as exc:  # noqa: BLE001
                text = ""
                tqdm.write(f"[warn] sample {idx} failed: {exc}")
            finally:
                for im in images[len(few_shot_imgs):]:
                    try:
                        im.close()
                    except Exception:
                        pass
            fw.write(json.dumps({
                "index": idx, "uid": ex.get("uid"), "question": q,
                "contextual_features": cf, "prompt": prompt,
                "prediction": text, "reference": ex.get("output", ""),
            }, ensure_ascii=False) + "\n")
    print(f"Saved to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
