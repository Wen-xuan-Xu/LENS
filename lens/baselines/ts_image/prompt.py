# -*- coding: utf-8 -*-
"""Prompt templates for the TS-Image baseline (Qwen2.5-VL on rendered charts).

The series are plotted as a multi-panel chart image (see ``render_plots.py`` /
the ``--plots-dir`` argument of ``query_vlm.py``); the VLM is then asked for a
clinical narrative (``PROMPT_TEMPLATE_NARRATIVE``) or a single answer
(``PROMPT_TEMPLATE_QA``). ``{contextual_features}`` is filled with the
sleep-duration / conversation-length summary lines; ``{question}`` with the QA
question.
"""

PROMPT_TEMPLATE_NARRATIVE = """You are a clinical psychologist interpreting behavioral and physiological data visualized in the image below.
Each chart represents one data stream collected continuously during the past four hours:

1. Heart rate (bpm): indicator of arousal, stress, and autonomic balance.
2. Pseudoactigraphy: derived from wrist accelerometer signals, representing movement intensity and rest-activity rhythm.
3. Steps per minute: reflects overall mobility and engagement in physical activity.
4. Stress level: Garmin HRV-based estimation of physiological stress.
5-6. GPS coordinates (longitude and latitude): capture spatial mobility and time spent in different environments.
7. Phone unlock status: number of unlock events per minute, representing cognitive or social engagement.

Additional contextual features:
{contextual_features}

Your task:
Analyze the figure carefully and provide a clinical summary of the user's recent psychological and behavioral state,
as if you were writing a short report based on PHQ-9-related observations.

In your reasoning, consider: interest or pleasure in activities; mood (sadness, hopelessness); energy and fatigue;
sleep quality; appetite or eating changes; self-evaluation (guilt, self-worth); concentration or attention;
psychomotor changes (slowed or restless behavior); anxiety or worry.

IMPORTANT INSTRUCTIONS:
- Write a single, concise clinical narrative (150-200 words maximum).
- Do NOT generate multiple versions, drafts, or revisions; do NOT repeat content.
- Avoid mentioning charts, axes, or numerical values.
- Write naturally, as if summarizing for another clinician. Stop immediately after the summary.
"""

PROMPT_TEMPLATE_QA = """You are a clinical psychologist interpreting behavioral and physiological data shown in the accompanying image.
The visualization contains seven streams collected across the past four hours:

1. Heart rate (bpm): indicator of arousal, stress, and autonomic balance.
2. Pseudoactigraphy: derived from wrist accelerometer signals, representing movement intensity and rest-activity rhythm.
3. Steps per minute: reflects overall mobility and engagement in physical activity.
4. Stress level: Garmin HRV-based estimation of physiological stress.
5-6. GPS coordinates (longitude and latitude): capture spatial mobility and time spent in different environments.
7. Phone unlock status: number of unlock events per minute, representing cognitive or social engagement.

Additional contextual features:
{contextual_features}

Question:
{question}

Your task:
Analyze the figure and answer the question directly. Base your reasoning only on observable behavioral and
physiological patterns plus the contextual features. Produce a concise, clinically grounded answer (2-3 sentences)
with no bullet points.

IMPORTANT INSTRUCTIONS:
- Provide one clear answer, no alternative scenarios.
- Avoid referencing charts, axes, or specific numeric values; do not speculate beyond the available evidence.
- Stop immediately after giving the answer.
"""
