"""Configuration helpers for the voice-guided grasping demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = REPO_ROOT / "config" / "grasp_profile.yaml"
DEFAULT_ALIASES = REPO_ROOT / "config" / "voice_aliases_zh.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML document and return an empty dictionary for empty files."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_repo_path(path: str | Path, base: Path | None = None) -> Path:
    """Resolve a possibly relative path against the repository root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base or REPO_ROOT).joinpath(candidate).resolve()
