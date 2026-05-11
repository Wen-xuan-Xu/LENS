"""Rule-based EMA -> narrative template generator.

Converts the 14-item EMA self-report (PHQ-9 + GAD-derived + a negative-event
question), scored on a 0-100 scale, into deterministic, item-level and
summary-level narrative templates.  These templates are the input to the LLM
rewriting stage (Qwen2.5-14B) and ultimately the ground-truth labels used to
train LENS.

EMA discretization (paper Appendix B):
    0-25  -> "not at all"
    26-50 -> "sometimes"
    51-75 -> "often"
    76-100-> "constantly"
Q3 (sleep, refers to "last night") uses an intensity scale instead:
    0-25 minimal / 26-50 moderate / 51-75 significant / 76-100 severe.

PHQ-9 total: each Q1-Q9 0-100 value is mapped to the standard 0-3 quartiles
and summed, then interpreted with the usual severity bands.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class EMATemplateGenerator:
    """Rule-based template generator for EMA data with 14 questions."""

    def __init__(self) -> None:
        # Frequency adverb mapping on the 0-100 scale (four equal intervals).
        self.frequency_map = {
            (0, 25): "not at all",
            (26, 50): "sometimes",
            (51, 75): "often",
            (76, 100): "constantly",
        }

        # Intensity mapping for Q3 (sleep, refers to last night specifically).
        self.sleep_intensity_map = {
            (0, 25): "minimal",
            (26, 50): "moderate",
            (51, 75): "significant",
            (76, 100): "severe",
        }

        # Template for each question, using frequency adverbs consistently.
        self.question_templates = {
            # PHQ-9 core questions (Q1-Q9) plus the somatic item (Q10).
            "Q1": "The user {frequency} had little interest or pleasure in doing things",
            "Q2": "The user {frequency} felt down, depressed, or hopeless",
            "Q3": "The user had {intensity} trouble with sleep last night",
            "Q4": "The user {frequency} felt tired or had little energy",
            "Q5": "The user {frequency} had poor appetite or was overeating",
            "Q6": "The user {frequency} felt bad about themselves",
            "Q7": "The user {frequency} had trouble concentrating",
            "Q8": "The user {frequency} moved or spoke slowly, or was fidgeting more",
            "Q9": "The user {frequency} had thoughts of hurting themselves or that they "
                  "would be better off dead",
            "Q10": "The user {frequency} had headaches, abdominal discomfort, or body aches",
            # Q11 comprehension questions (positive framing).
            "Q11(0)": "The user {frequency} had interest or pleasure in doing things",
            "Q11(1)": "The user {frequency} had a lot of energy",
            "Q11(2)": "The user {frequency} was able to concentrate well",
            # Anxiety questions (Q12-Q13).
            "Q12": "The user {frequency} felt nervous, anxious, or on edge",
            "Q13": "The user {frequency} was not able to stop or control worrying",
            # Negative event (Q14) - handled separately as Yes/No.
            "Q14": "The user experienced a negative event",
        }

        # PHQ-9 core questions for total-score calculation (with ema_ prefix).
        self.phq9_questions = ["ema_Q1", "ema_Q2", "ema_Q3", "ema_Q4", "ema_Q5",
                               "ema_Q6", "ema_Q7", "ema_Q8", "ema_Q9"]

        # All questions in order (with ema_ prefix matching the feature-rows CSV).
        self.all_questions = ["ema_Q1", "ema_Q2", "ema_Q3", "ema_Q4", "ema_Q5",
                              "ema_Q6", "ema_Q7", "ema_Q8", "ema_Q9", "ema_Q10",
                              "ema_Q11(0)", "ema_Q11(1)", "ema_Q11(2)", "ema_Q12",
                              "ema_Q13", "ema_Q14"]

        # PHQ-9 severity interpretation (standard ranges).
        self.phq9_severity = {
            (0, 0): "No depression",
            (1, 4): "Minimal depression",
            (5, 9): "Mild depression",
            (10, 14): "Moderate depression",
            (15, 19): "Moderately severe depression",
            (20, 27): "Severe depression",
        }

    # --------------------------------------------------------------------- #
    # Discretization helpers
    # --------------------------------------------------------------------- #
    def get_frequency_adverb(self, score: float) -> str:
        """Map a 0-100 score to a frequency adverb."""
        if pd.isna(score) or score < 0:
            return "not at all"
        for (min_val, max_val), frequency in self.frequency_map.items():
            if min_val <= score <= max_val:
                return frequency
        return "unknown"

    def get_sleep_intensity(self, score: float) -> str:
        """Map a 0-100 score to an intensity word for Q3 (sleep)."""
        if pd.isna(score) or score < 0:
            return "no"
        for (min_val, max_val), intensity in self.sleep_intensity_map.items():
            if min_val <= score <= max_val:
                return intensity
        return "unknown"

    def calculate_phq9_score(self, scores: Dict) -> int:
        """PHQ-9 total: convert each 0-100 item to 0-3 quartiles and sum Q1-Q9."""
        total = 0
        for q in self.phq9_questions:
            if q in scores:
                score_100 = scores[q]
                if pd.isna(score_100):
                    continue
                if 0 <= score_100 <= 25:
                    score_3 = 0  # not at all
                elif 26 <= score_100 <= 50:
                    score_3 = 1  # several days
                elif 51 <= score_100 <= 75:
                    score_3 = 2  # more than half the days
                else:  # 76-100
                    score_3 = 3  # nearly every day
                total += score_3
        return total

    def get_phq9_interpretation(self, total_score: int) -> str:
        """Get the PHQ-9 severity interpretation for a total score."""
        for (min_val, max_val), interpretation in self.phq9_severity.items():
            if min_val <= total_score <= max_val:
                return interpretation
        return "Score out of range"

    # --------------------------------------------------------------------- #
    # Single-question / parsing helpers
    # --------------------------------------------------------------------- #
    def generate_question_sentence(self, question: str, score: Any) -> str:
        """Generate a template sentence for a single question."""
        if question == "Q14":
            return self.question_templates["Q14"] if str(score).lower() == "yes" else ""
        template = self.question_templates.get(question, "")
        if not template:
            return ""
        if question == "Q3":
            return template.format(intensity=self.get_sleep_intensity(float(score)))
        return template.format(frequency=self.get_frequency_adverb(float(score)))

    def parse_ema_row(self, row: pd.Series) -> Dict:
        """Pull the ema_Q* columns and metadata out of a feature-rows row."""
        scores = {q: row[q] for q in self.all_questions if q in row}
        return {
            "meta": {
                "timestamp": row.get("response_time", ""),
                "uid": row.get("uid", ""),
            },
            "scores": scores,
        }

    # --------------------------------------------------------------------- #
    # Narrative / per-question generation
    # --------------------------------------------------------------------- #
    def generate_narrative(self, parsed_data: Dict) -> str:
        """Generate the summary-level narrative as one coherent passage."""
        scores = parsed_data["scores"]
        symptoms: List[str] = []

        standard_questions = ["ema_Q1", "ema_Q2", "ema_Q4", "ema_Q5", "ema_Q6",
                              "ema_Q7", "ema_Q8", "ema_Q9", "ema_Q10"]
        for question in standard_questions:
            if question in scores and not pd.isna(scores[question]):
                template = self.question_templates.get(question.replace("ema_", ""), "")
                if template:
                    frequency = self.get_frequency_adverb(scores[question])
                    if frequency != "unknown":
                        symptoms.append(template.format(frequency=frequency))

        if "ema_Q3" in scores and not pd.isna(scores["ema_Q3"]):
            template = self.question_templates.get("Q3", "")
            if template:
                intensity = self.get_sleep_intensity(scores["ema_Q3"])
                if intensity != "unknown":
                    symptoms.append(template.format(intensity=intensity))

        for question in ["ema_Q11(0)", "ema_Q11(1)", "ema_Q11(2)", "ema_Q12", "ema_Q13"]:
            if question in scores and not pd.isna(scores[question]):
                template = self.question_templates.get(question.replace("ema_", ""), "")
                if template:
                    frequency = self.get_frequency_adverb(scores[question])
                    if frequency != "unknown":
                        symptoms.append(template.format(frequency=frequency))

        if "ema_Q14" in scores and str(scores["ema_Q14"]).lower() == "yes":
            symptoms.append(self.question_templates["Q14"])

        phq9_total = self.calculate_phq9_score(scores)
        phq9_interp = self.get_phq9_interpretation(phq9_total)
        symptoms.append(f"Overall, the user presented with {phq9_interp.lower()}")

        return ". ".join(symptoms) + "."

    def generate_question_responses(self, parsed_data: Dict) -> Dict:
        """Generate an individual response sentence for each answered question."""
        scores = parsed_data["scores"]
        responses: Dict[str, str] = {}
        for question in self.all_questions:
            if question not in scores or pd.isna(scores[question]):
                continue
            q_key = question.replace("ema_", "")
            if q_key == "Q14":
                responses[question] = (self.question_templates["Q14"]
                                       if str(scores[question]).lower() == "yes"
                                       else "The user did not experience a negative event")
                continue
            template = self.question_templates.get(q_key, "")
            if not template:
                continue
            if q_key == "Q3":
                intensity = self.get_sleep_intensity(scores[question])
                if intensity != "unknown":
                    responses[question] = template.format(intensity=intensity)
            else:
                frequency = self.get_frequency_adverb(scores[question])
                if frequency != "unknown":
                    responses[question] = template.format(frequency=frequency)
        return responses

    def process_single_entry(self, data_row: Dict) -> str:
        """Convenience: parse a dict row and return the summary narrative."""
        return self.generate_narrative(self.parse_ema_row(pd.Series(data_row)))

    # --------------------------------------------------------------------- #
    # Batch processing over a feature-rows CSV
    # --------------------------------------------------------------------- #
    def process_csv(self, input_file: str, output_file_templates: str,
                    output_file_sampled: str, n_sampled: int = 2,
                    seed: int | None = None) -> None:
        """Generate item + summary templates for every row of ``input_file``.

        Writes two JSON files:
          * ``output_file_templates``: one record per EMA with ``template_narrative``.
          * ``output_file_sampled``: one record per EMA with ``question_responses``
            for a random subset of the answered items.
        """
        rng = random.Random(seed)
        df = pd.read_csv(input_file)
        if "match_success" in df.columns:
            df = df[df["match_success"] == True]  # noqa: E712

        Path(output_file_templates).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file_sampled).parent.mkdir(parents=True, exist_ok=True)

        templates_results: List[Dict] = []
        sampled_results: List[Dict] = []
        for _, row in df.iterrows():
            parsed = self.parse_ema_row(row)
            templates_results.append({
                "uid": row.get("uid", ""),
                "ema_timestamp": row.get("response_time", ""),
                "template_narrative": self.generate_narrative(parsed),
            })
            ema_answers = {c: row[c] for c in df.columns if c.startswith("ema_Q")}
            non_null = [q for q, v in ema_answers.items() if pd.notna(v)]
            picked = rng.sample(non_null, k=min(n_sampled, len(non_null))) if non_null else []
            all_resp = self.generate_question_responses(parsed)
            sampled_results.append({
                "uid": row.get("uid", ""),
                "ema_timestamp": row.get("response_time", ""),
                "question_responses": {q: all_resp[q] for q in picked if q in all_resp},
            })

        with open(output_file_templates, "w", encoding="utf-8") as f:
            json.dump(templates_results, f, ensure_ascii=False, indent=2)
        with open(output_file_sampled, "w", encoding="utf-8") as f:
            json.dump(sampled_results, f, ensure_ascii=False, indent=2)
