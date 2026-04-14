from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ModelRegistryError(RuntimeError):
    pass


REQUIRED_FIELDS = {"name", "path", "sha256", "architecture", "exported_at"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    if not registry_path.exists():
        raise ModelRegistryError(f"Model registry not found: {registry_path}")

    with registry_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ModelRegistryError("Model registry must be a JSON object")

    return data


def validate_registry(path: str | Path, strict: bool = True) -> dict[str, Any]:
    registry = load_registry(path)
    registry_path = Path(path)

    models = registry.get("models")
    if not isinstance(models, dict):
        raise ModelRegistryError("Model registry must define 'models' as an object")

    for model_name, metadata in models.items():
        if not isinstance(metadata, dict):
            raise ModelRegistryError(f"Registry entry for {model_name} must be an object")

        missing = REQUIRED_FIELDS - metadata.keys()
        if missing:
            raise ModelRegistryError(
                f"{model_name} registry entry missing fields: {sorted(missing)}"
            )

        model_path = Path(str(metadata.get("path", f"models/{model_name}.pth")))
        if not model_path.is_absolute():
            candidate_from_root = registry_path.parent.parent / model_path
            candidate_from_models = registry_path.parent / model_path
            if candidate_from_root.exists() or not candidate_from_models.exists():
                model_path = candidate_from_root
            else:
                model_path = candidate_from_models

        expected = str(metadata.get("sha256", ""))

        if strict and isinstance(expected, str) and expected.startswith("PENDING_"):
            raise ModelRegistryError(f"{model_name} has unresolved placeholder hash in strict mode")

        if not model_path.exists():
            if strict:
                raise ModelRegistryError(f"Missing model file for {model_name}: {model_path}")
            continue

        if expected and not str(expected).startswith("PENDING_") and expected != "REPLACE_WITH_REAL_SHA256":
            actual = _sha256(model_path)
            if actual.lower() != expected.lower():
                raise ModelRegistryError(
                    f"Checksum mismatch for {model_name}: expected {expected}, got {actual}"
                )

    return models
