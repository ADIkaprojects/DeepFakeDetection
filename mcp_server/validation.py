from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import validate


class SchemaValidationError(RuntimeError):
    pass


def load_schemas(schema_dir: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(schema_dir)
    mapping: dict[str, dict[str, Any]] = {}
    for schema_file in root.glob("*.schema.json"):
        method = schema_file.name.replace(".schema.json", "")
        mapping[method] = json.loads(schema_file.read_text(encoding="utf-8"))
    return mapping


def validate_payload(method: str, payload: dict[str, Any], schemas: dict[str, dict[str, Any]]) -> None:
    schema = schemas.get(method)
    if schema is None:
        raise SchemaValidationError(f"No schema configured for method: {method}")

    try:
        validate(instance=payload, schema=schema)
    except Exception as exc:
        raise SchemaValidationError(str(exc)) from exc
