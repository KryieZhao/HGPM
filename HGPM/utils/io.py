from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PACKAGE_ROOT.parent


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_config(config_path: str | Path, output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    shutil.copy2(config_path, output_dir / "config_used.yaml")


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a repo-relative path against the package root first.

    Falls back to the parent directory so that the package can also be
    consumed from a wrapping workspace where shared data lives one level up.
    Absolute paths are returned unchanged.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    packaged = PACKAGE_ROOT / candidate
    if packaged.exists():
        return packaged

    workspace = WORKSPACE_ROOT / candidate
    if workspace.exists():
        return workspace

    return packaged
