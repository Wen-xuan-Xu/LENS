"""LENS: LLM-Enabled Narrative Synthesis for Mental Health.

See https://github.com/Wen-xuan-Xu/LENS and the accompanying paper.
This package is organized into four code groups:
  - lens.feature_engineering : raw passive-sensing + EMA -> windowed feature rows
  - lens.data_pipeline       : EMA -> rule templates -> LLM enrichment -> multi-judge QC -> training datasets
  - lens.training            : two-stage SFT of the LENS time-series-to-text model (via the ChatTS-Training submodule (`./ChatTS-Training`))
  - lens.eval                : inference + NLP metrics + LLM-as-judge clinical alignment
  - lens.baselines           : TS-Text (few-shot), TS-Image (VLM), TS-Text-FT (text-only fine-tune)
"""

__version__ = "0.1.0"
