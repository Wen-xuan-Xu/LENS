"""LENS data-synthesis pipeline.

EMA self-reports -> rule-based item + summary template narratives -> LLM
rewriting (Qwen2.5-14B) -> multi-agent LLM-as-a-judge QC -> HF Arrow datasets
-> jsonl with ``<ts><ts/>`` placeholders consumed by the training stack.
"""
