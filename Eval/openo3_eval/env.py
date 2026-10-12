"""Centralized environment variable constants for Open-o3 Eval.

All values are read from ``os.environ`` at import time (after ``load_dotenv()``
in ``__main__``).  Downstream code should ``from openo3_eval.env import X``.

Per-dataset env vars (e.g. ``VIDEO_MME_SAMPLE_FPS``) are resolved in
``BaseDatasetPP`` helpers — this module only exposes the global defaults.

Naming convention:
  - Global: ``OPENO3_EVAL_<NAME>``
  - Per-dataset: ``<DATASET>_<NAME>``  (e.g. ``VIDEO_MME_SAMPLE_FPS``)
"""

from __future__ import annotations

import os
from typing import Tuple


def _parse_resize(val: str) -> Tuple[int, int] | None:
    """Parse ``"WxH"`` → ``(W, H)`` or ``""`` / ``"none"`` → ``None``."""
    if not val or val.lower() == "none":
        return None
    parts = val.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid resize format: {val!r}, expected 'WxH' (e.g. '224x224')")
    return (int(parts[0]), int(parts[1]))


# ---------------------------------------------------------------------------
# Global constants (read once at import time)
# ---------------------------------------------------------------------------

CACHE_DIR: str = os.environ.get("OPENO3_EVAL_CACHE_DIR", ".openo3_eval_cache")

INPUT_MODE: str = os.environ.get("OPENO3_EVAL_INPUT_MODE", "video_path")
SAMPLE_FPS: float = float(os.environ.get("OPENO3_EVAL_SAMPLE_FPS", 2.0))
RESIZE: Tuple[int, int] | None = _parse_resize(os.environ.get("OPENO3_EVAL_RESIZE", ""))
JPEG_QUALITY: int = int(os.environ.get("OPENO3_EVAL_JPEG_QUALITY", 95))

DECORD_TIMEOUT: int = int(os.environ.get("OPENO3_EVAL_DECORD_TIMEOUT", 300))
EXTRACT_WORKERS: int = int(os.environ.get("OPENO3_EVAL_EXTRACT_WORKERS", min(32, os.cpu_count() or 32)))
