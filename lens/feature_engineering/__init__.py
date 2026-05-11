"""LENS feature engineering: passive-sensing preprocessing + EMA codebook.

Raw sensor streams + EMA self-reports -> ``feature_rows.csv`` (per-EMA sensor
windows) -> ``filtered_feature_rows.csv`` (adds the per-minute phone-unlock
column and the rule-based / LLM narrative text).

Submodules:
- ``build_feature_rows``      : canonical raw -> feature_rows builder
- ``add_unlock_and_narratives``: adds unlock column + narrative text
- ``extract_enhanced_fields`` : slim the enriched JSONL to the rewritten text
- ``data_integrity_check``    : assert per-channel window lengths
- ``ema_codebook``            : EMA item inventory + discretisation + scoring
- ``legacy.ema_sensor_matcher``: older, superseded matcher (reference only)
"""

__all__ = [
    "ema_codebook",
    "build_feature_rows",
    "add_unlock_and_narratives",
    "extract_enhanced_fields",
    "data_integrity_check",
]
