"""Repository-relative path helpers."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def get_project_root() -> Path:
    """Return the repository root, optionally overridden by LDM_PROJECT_ROOT."""
    override = os.environ.get("LDM_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT


def project_path(*parts: str | os.PathLike[str]) -> Path:
    """Build an absolute path inside the repository."""
    return get_project_root().joinpath(*parts)


def data_path(*parts: str | os.PathLike[str]) -> Path:
    """Build a path below the data root."""
    root = os.environ.get("LDM_DATA_ROOT")
    data_root = Path(root).expanduser().resolve() if root else project_path("Data")
    return data_root.joinpath(*parts)


def model_path(*parts: str | os.PathLike[str]) -> Path:
    """Build a path below the trained-model root."""
    root = os.environ.get("LDM_MODEL_ROOT")
    model_root = Path(root).expanduser().resolve() if root else project_path("Trained_Models")
    return model_root.joinpath(*parts)


def _coerce_relative_path(path: str | os.PathLike[str]) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return project_path(path_obj)


def resolve_input_path(env_var: str, default_relative_path: str | os.PathLike[str]) -> Path:
    """Resolve an input path from an environment variable or repo-relative default."""
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser().resolve()
    return _coerce_relative_path(default_relative_path)


def resolve_output_path(
    relative_path: str | os.PathLike[str],
    env_var: str = "LDM_OUTPUT_DIR",
) -> Path:
    """Resolve an output path and create its parent directory.

    If env_var is set, the path is rooted there; otherwise it is rooted in the
    current repository checkout.
    """
    output_root = os.environ.get(env_var)
    path_obj = Path(relative_path)
    if output_root and not path_obj.is_absolute():
        resolved = Path(output_root).expanduser().resolve() / path_obj
    else:
        resolved = _coerce_relative_path(path_obj)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_parent(path: str | os.PathLike[str]) -> Path:
    """Resolve a path and create its parent directory."""
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
