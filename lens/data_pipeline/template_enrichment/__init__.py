"""Rule-based EMA template generation + LLM narrative enrichment."""

from .generator import EMATemplateGenerator
from .prompt import create_enrichment_prompt

__all__ = ["EMATemplateGenerator", "create_enrichment_prompt"]

