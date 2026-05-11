"""Judge-ensemble configuration and evaluation rubric.

The multi-agent LLM-as-a-judge filter (paper Section 3.1) scores each
LLM-rewritten narrative against its rule-based template on five 1-5 Likert
dimensions -- Factual Alignment, Symptom Coverage, Severity Fidelity,
Fluency & Naturalness, Hallucination Risk -- using an ensemble of three small
open models (Mistral-7B, Llama-3.1-8B, Qwen2.5-7B).  Confidence-weighted
voting across the ensemble yields a PASS/FAIL decision per narrative.

Model paths and service URLs are read from environment variables so no local
filesystem layout or endpoint is baked in.  Defaults assume each model is
served locally via SGLang on ports 30000-30002.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List
from urllib.parse import urlparse


@dataclass
class ModelConfig:
    """Configuration for a single judge-model deployment."""

    model_name: str
    model_path: str  # HF Hub id or local path; set via env var.
    service_url: str  # e.g. "http://127.0.0.1:30000"
    extra_args: str = ""
    weight: float = 1.0  # Voting weight based on model capability.
    gpu_id: int = 0  # GPU id this model is pinned to when self-deployed.

    @property
    def host(self) -> str:
        return urlparse(self.service_url).hostname or "127.0.0.1"

    @property
    def port(self) -> int:
        return urlparse(self.service_url).port or 30000


def _default_judge_models() -> List[ModelConfig]:
    """Build the default judge ensemble from environment variables.

    Env vars (all optional):
      LENS_JUDGE_QWEN_PATH / _LLAMA_PATH / _MISTRAL_PATH  -- model id or path
      LENS_JUDGE_QWEN_URL  / _LLAMA_URL  / _MISTRAL_URL   -- service URL
    """
    return [
        ModelConfig(
            model_name="qwen2.5-7b",
            model_path=os.environ.get("LENS_JUDGE_QWEN_PATH", "Qwen/Qwen2.5-7B-Instruct"),
            service_url=os.environ.get("LENS_JUDGE_QWEN_URL", "http://127.0.0.1:30000"),
            weight=1.0, gpu_id=1,
        ),
        ModelConfig(
            model_name="llama3-8b",
            model_path=os.environ.get("LENS_JUDGE_LLAMA_PATH", "meta-llama/Llama-3.1-8B-Instruct"),
            service_url=os.environ.get("LENS_JUDGE_LLAMA_URL", "http://127.0.0.1:30001"),
            weight=1.0, gpu_id=2,
        ),
        ModelConfig(
            model_name="mistral-7b",
            model_path=os.environ.get("LENS_JUDGE_MISTRAL_PATH", "mistralai/Mistral-7B-Instruct-v0.3"),
            service_url=os.environ.get("LENS_JUDGE_MISTRAL_URL", "http://127.0.0.1:30002"),
            weight=1.0, gpu_id=3,
        ),
    ]


JUDGE_MODELS_AVAILABLE: List[ModelConfig] = _default_judge_models()


@dataclass
class VotingConfig:
    """Configuration for the consensus-voting step."""

    voting_method: str = "weighted_median"  # weighted_median | majority | average
    min_agreement_threshold: float = 0.6
    tie_breaking_model: str = "qwen2.5-7b"
    confidence_weighting: bool = True
    outlier_detection: bool = True
    pass_threshold: int = 20  # PASS if combined score (out of 25) >= this.


# ============================================================================
# EVALUATION RUBRIC & PROMPTS
# ============================================================================

JUDGE_SYSTEM_PROMPT = """As a highly meticulous and objective clinical quality reviewer, your primary responsibility is to evaluate the quality and safety of an AI-generated mental health narrative.
You must ground your judgment strictly in the provided source data.

You will score the "AI-generated Narrative" on five specific dimensions using a 1-5 Likert scale and then provide a concise, structured critique.

Template-Based Narrative (Baseline for Comparison/ground):
This is the rule-based description generated directly from the EMA self-report scores using a fixed template. Use this as a baseline for factual and coverage assessment.

AI-generated Narrative (To Be Evaluated):
This is the AI-rewrite version of the narrative that you must score and critique.

You must respond with ONLY a valid JSON object. No additional text before or after."""

JUDGE_USER_PROMPT_TEMPLATE = """Template-Based Narrative (Baseline for Comparison/ground):
{original_template}

AI-generated Narrative (To Be Evaluated):
{enriched_narrative}

Please evaluate the AI-generated Narrative based on the following five dimensions. For each dimension, provide a score from 1 (Very Poor) to 5 (Excellent).

- Factual Alignment: Does the narrative accurately reflect the presence or absence of symptoms reported in the Template-Based Narrative? Does it contradict any facts from the source data?
Scoring Guide:
    1: Contains significant factual contradictions (e.g., states a symptom is present when it was scored 0).
    3: Generally aligns, but may contain minor, subtle inaccuracies or misinterpretations.
    5: Perfectly aligns with the source data, with no factual errors or contradictions.

- Symptom Coverage: Does the narrative mention or allude to all the relevant symptoms that were reported with a non-zero score in the Template-Based Narrative?
Scoring Guide:
    1: Misses multiple significant, reported symptoms.
    3: Covers the most severe symptoms but misses one or two less severe ones.
    5: Comprehensively covers all reported symptoms, giving appropriate attention to each.

-Severity Fidelity: Does the language and tone used to describe each symptom accurately reflect its severity level (e.g., 'not at all', 'sometimes', 'often', 'constantly') from the Template-Based Narrative?
Scoring Guide:
    1: Grossly misrepresents the severity of symptoms
    3: Captures the general sense of severity but lacks precision or occasionally over/understates it.
    5: The language precisely and appropriately conveys the specific severity level for each symptom.

- Fluency & Naturalness: Is the narrative well-written, coherent, and natural-sounding? Does it avoid the robotic, repetitive language of the Template-Based Narrative without sounding overly dramatic or artificial?
Scoring Guide:
    1: Awkward, disjointed, or sounds highly artificial and machine-generated.
    3: Readable and generally fluent, but may have some slightly unnatural phrasing or repetition.
    5: Flows naturally and reads like it could have been written by a person. It is engaging and easy to understand.

-Hallucination Risk
Question: Does the narrative introduce any new symptoms, details, or assumptions that are NOT supported by the Template-Based Narrative (This is a safety-critical dimension).
Scoring Guide:
    1: Introduces significant and potentially harmful fabrications (e.g., mentions suicidal ideation when not reported, or invents specific life events).
    3: Adds minor, unsupported details that are clinically neutral but still outside the provided data (e.g., "feeling tired in the morning" when only "fatigue" was reported).
    5: Strictly adheres to the information given in the source data. No information is invented or fabricated.

CONFIDENCE SCORING GUIDE:
For each dimension, provide a confidence score from 0.0 to 1.0 indicating how certain you are about your evaluation:
- 1.0: Completely certain - The evidence is crystal clear and unambiguous
- 0.8-0.9: High confidence - Strong evidence supports the score with minimal doubt
- 0.6-0.7: Moderate confidence - Reasonable evidence but some room for interpretation
- 0.4-0.5: Low confidence - Limited evidence or significant ambiguity in the text
- 0.1-0.3: Very low confidence - Minimal evidence or highly uncertain evaluation
- 0.0: No confidence - Unable to make a reliable assessment

RETURN your evaluation result in the following JSON format:

{{
  "scores": [Factual Alignment, Symptom Coverage, Severity Fidelity, Fluency & Naturalness, Hallucination Risk],
  "confidence": [confidence_1, confidence_2, confidence_3, confidence_4, confidence_5],
  "critique": {{
    "Factual Alignment": "Brief evaluation.",
    "Symptom Coverage": "Brief evaluation.",
    "Severity Fidelity": "Brief evaluation.",
    "Fluency & Naturalness": "Brief evaluation.",
    "Hallucination Risk": "Brief evaluation."
  }}
}}

"""
