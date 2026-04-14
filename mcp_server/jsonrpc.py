from __future__ import annotations

from typing import Any


def jsonrpc_response(result: Any, request_id: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def jsonrpc_error(message: str, request_id: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32000, "message": message},
    }
