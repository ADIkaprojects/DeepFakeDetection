import pytest

from mcp_server.validation import SchemaValidationError, load_schemas, validate_payload


def test_face_detector_schema_accepts_valid_payload() -> None:
    schemas = load_schemas("mcp_server/schemas")
    validate_payload("face_detector", {"frame_b64": "abc"}, schemas)


def test_face_detector_schema_rejects_invalid_payload() -> None:
    schemas = load_schemas("mcp_server/schemas")
    with pytest.raises(SchemaValidationError):
        validate_payload("face_detector", {"bad": "payload"}, schemas)
