"""Training-data build chain: enriched JSONL -> Arrow DatasetDict -> jsonl."""

FINAL_TS_ORDER = [
    "hr_window", "zcr_prod", "steps_first", "stress_window",
    "gps_lon", "gps_lat", "unlock",
]
FINAL_TS_LENGTHS = [1440, 480, 240, 240, 24, 24, 240]

__all__ = ["FINAL_TS_ORDER", "FINAL_TS_LENGTHS"]

