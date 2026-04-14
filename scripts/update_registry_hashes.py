from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("D:/deepfake_detection")
REGISTRY_PATH = BASE / "models" / "registry.json"

MODEL_FILES = {
    "reface_atn": BASE / "models" / "reface_atn.pth",
    "deepsafe": BASE / "models" / "deepsafe.pth",
}

REQUIRED_FIELDS = ("name", "path", "sha256", "architecture", "exported_at")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_path_for(model_key: str) -> str:
    model_path = MODEL_FILES.get(model_key)
    if model_path is None:
        return f"models/{model_key}.pth"
    return f"models/{model_path.name}"


def _normalize_entry(model_key: str, entry: dict[str, object]) -> dict[str, str]:
    now_iso = datetime.now(timezone.utc).isoformat()
    normalized: dict[str, str] = {
        "name": str(entry.get("name") or model_key),
        "path": str(entry.get("path") or _default_path_for(model_key)),
        "sha256": str(entry.get("sha256") or ""),
        "architecture": str(entry.get("architecture") or "unknown"),
        "exported_at": str(entry.get("exported_at") or now_iso),
    }
    return normalized


def validate_registry(path: str | Path) -> list[str]:
    """Validate canonical registry entry schema and return human-readable violations."""
    registry_path = Path(path)
    violations: list[str] = []

    if not registry_path.exists():
        return [f"registry missing: {registry_path}"]

    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"registry parse error: {exc}"]

    models = registry.get("models") if isinstance(registry, dict) else None
    if not isinstance(models, dict):
        return ["registry field 'models' must be an object"]

    for model_key, raw_entry in models.items():
        if not isinstance(raw_entry, dict):
            violations.append(f"{model_key}: entry must be an object")
            continue
        for field in REQUIRED_FIELDS:
            value = raw_entry.get(field)
            if not isinstance(value, str) or not value.strip():
                violations.append(f"{model_key}: missing or empty field '{field}'")

    return violations


def update_registry_hashes(registry_path: Path = REGISTRY_PATH) -> tuple[int, list[str]]:
    if not registry_path.exists():
        raise FileNotFoundError(f"registry.json not found at {registry_path}")

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise ValueError("registry.json must be a JSON object")

    models_raw = registry.get("models", {})
    models: dict[str, dict[str, object]]

    if isinstance(models_raw, list):
        models = {}
        for entry in models_raw:
            if isinstance(entry, dict):
                model_name = str(entry.get("name") or "")
                if model_name:
                    models[model_name] = entry
    elif isinstance(models_raw, dict):
        models = {k: v for k, v in models_raw.items() if isinstance(v, dict)}
    else:
        raise ValueError("registry.json has unexpected format: 'models' must be a list or dict")

    updated = 0
    for model_key, model_path in MODEL_FILES.items():
        current = models.get(model_key, {})
        entry = _normalize_entry(model_key, current)
        entry["path"] = _default_path_for(model_key)

        if model_path.exists():
            digest = sha256(model_path)
            entry["sha256"] = digest
            updated += 1
            print(f"[SHA256] {model_key}: {digest}")
        else:
            print(f"[SHA256] FAIL {model_key}: file not found at {model_path}")

        models[model_key] = entry

    registry["models"] = models
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")

    violations = validate_registry(registry_path)
    return updated, violations


def main() -> int:
    try:
        updated, violations = update_registry_hashes(REGISTRY_PATH)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"[registry.json] FAIL {exc}")
        return 1

    print(f"\n[registry.json] Updated entries: {updated}")
    if violations:
        print("[registry.json] Schema violations detected:")
        for violation in violations:
            print(f"- {violation}")
        return 1

    print("[registry.json] Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
