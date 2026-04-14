import json

from mcp_server.jsonrpc import jsonrpc_response


def test_jsonrpc_response_structure() -> None:
    response = jsonrpc_response({"ok": True}, 123)
    payload = json.loads(json.dumps(response))

    assert payload["jsonrpc"] == "2.0"
    assert payload["id"] == 123
    assert payload["result"]["ok"] is True
