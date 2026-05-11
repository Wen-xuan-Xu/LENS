"""System / user prompt templates for the LLM narrative-rewriting stage.

The rule-based templates produced by :class:`EMATemplateGenerator` are
mechanical and repetitive.  This stage asks an instruction-tuned LLM
(Qwen2.5-14B in the paper) to rewrite each template into a fluent narrative
*without* adding, removing, or altering any factual / severity content.  The
rewritten narratives become the ground-truth training labels.
"""
from __future__ import annotations

import json
from typing import List

SYSTEM_PROMPT = """You are both a mental health and language specialist experienced with clinical records concerning mental health conditions.
Your objective:
Rewrite rule-based psychological assessment templates into fluent, engaging narrative passages, strictly preserving every factual detail and original severity description. These improved narratives will act as ground truth labels for training AI models to predict mental health states using physiological sensor data (such as smartwatch readings).
Paraphrasing guidelines:
1. Factual Accuracy and Preservation
- Retain every original severity level exactly as presented.
- Do not add any interpretations, clinical reasoning, or extra context.
- Preserve all specific data on frequency and intensity without omission or alteration.
2. Natural, Readable Language
- Remove mechanical or repetitive phrasing.
- Employ varied sentence structure and natural transitions between symptoms.
- Ensure the narrative flows as a natural human description would.
3. Consistency in Terminology and Tone
- Always use the same language for similar severity levels across all narratives.
- Maintain a uniform style and tone throughout all paraphrased outputs.
4. Accessibility and Clarity
- Write in straightforward, accessible language suitable for general audiences.
- Eliminate technical/clinical terms whenever possible.
- Use person-first, stigma-free wording.
- Make sentences clear and concise, yet thorough.
"""

USER_PROMPT_TEMPLATE = """Your task: Transform the below rule-based assessment into a well-structured, fluent narrative that fully preserves all factual content and improves readability.
Original Assessment:
{rule_based_template}
Enhanced Narrative:
"""


def create_enrichment_prompt(rule_based_template: str) -> List[dict]:
    """Build the chat-message list for narrative enrichment.

    Args:
        rule_based_template: the rule-based template text to rewrite.

    Returns:
        A list of ``{"role": ..., "content": ...}`` messages.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(rule_based_template=rule_based_template)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def create_batch_requests(jsonl_file: str, output_file: str, model: str = "qwen2.5-14b",
                          max_tokens: int = 300, temperature: float = 0.6) -> None:
    """Create a batched-request file (one JSON per line) from a templates JSONL.

    Each input line must contain a ``narrative`` field; the output file is
    suitable for an OpenAI-compatible ``/chat/completions`` batch endpoint.
    """
    requests: List[dict] = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            requests.append({
                "custom_id": f"request_{line_num}",
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": model,
                    "messages": create_enrichment_prompt(data["narrative"]),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.95,
                    "stop": ["**", "Input:", "Output:", "Example"],
                },
            })
    with open(output_file, "w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
