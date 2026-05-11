"""Path resolution helpers for the feature-engineering stage.

The private LENS study laid raw sensor CSVs out under
``<data_root>/raw_data/<channel>/...`` with sleep under
``<data_root>/raw_data/garmin_health_api/sleep/``.  The shipped synthetic
("fake") data uses ``<data_root>/raw_sensors/<channel>/...`` and
``<data_root>/garmin_health_api/sleep/``.  These helpers resolve either layout
so the same code runs against private and public inputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def raw_sensors_dir(data_root: Path) -> Path:
    """Return the directory that holds the per-channel sensor sub-directories."""
    data_root = Path(data_root)
    for candidate in ("raw_sensors", "raw_data"):
        p = data_root / candidate
        if p.is_dir():
            return p
    # Default to the public layout even if it does not exist yet.
    return data_root / "raw_sensors"


def sensor_channel_dir(data_root: Path, channel: str) -> Path:
    """Directory holding a single channel's CSVs (e.g. ``hr``, ``gps``)."""
    return raw_sensors_dir(data_root) / channel


def sleep_dir(data_root: Path) -> Path:
    """Directory holding the Garmin Health API sleep CSVs."""
    data_root = Path(data_root)
    candidates = [
        data_root / "garmin_health_api" / "sleep",
        data_root / "raw_data" / "garmin_health_api" / "sleep",
        raw_sensors_dir(data_root) / "garmin_health_api" / "sleep",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


def find_file(directory: Path, name: str) -> Optional[Path]:
    """Return ``directory / name`` if it exists, else ``None``."""
    p = Path(directory) / name
    return p if p.exists() else None
