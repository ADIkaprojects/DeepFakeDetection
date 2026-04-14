from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from aiohttp import web

from mcp_server.jsonrpc import jsonrpc_error, jsonrpc_response
from mcp_server.security import RateLimitConfig, SlidingWindowRateLimiter
from mcp_server.tools.deepfake_feedback import handle_deepfake_feedback, init_deepsafe
from mcp_server.tools.face_detector import handle_face_detector
from mcp_server.tools.frame_blender import handle_frame_blender
from mcp_server.tools.perturbation_generator import handle_perturbation_generator
from mcp_server.validation import SchemaValidationError, load_schemas, validate_payload
from src.blending.frame_blender import BlendConfig, FrameBlender
from src.detection.face_detector import FaceDetector, FaceDetectorConfig
from src.perturbation.atn_engine import ATNConfig, ATNEngine
from src.utils.config import load_config
from src.utils.logging_utils import configure_logging
from src.utils.model_registry import validate_registry

LOGGER = logging.getLogger("afs.mcp_server")
REQUIRED_REGISTRY_FIELDS = {"name", "path", "sha256", "architecture", "exported_at"}


@dataclass(slots=True)
class ServerState:
    detector: FaceDetector
    atn_engine: ATNEngine
    blender: FrameBlender
    schemas: dict[str, dict[str, Any]]
    api_key: str | None
    limiter: SlidingWindowRateLimiter | None


def build_dispatch(state: ServerState) -> dict[str, Callable[[dict], dict]]:
    return {
        "face_detector": lambda payload: handle_face_detector(state.detector, payload),
        "perturbation_generator": lambda payload: handle_perturbation_generator(state.atn_engine, payload),
        "frame_blender": lambda payload: handle_frame_blender(state.blender, payload),
        "deepfake_feedback": handle_deepfake_feedback,
    }


async def handle_jsonrpc(request: web.Request) -> web.Response:
    app_state: ServerState = request.app["state"]
    dispatch = build_dispatch(app_state)

    client_ip = request.remote or "unknown"
    if app_state.limiter and not app_state.limiter.allow(client_ip):
        return web.json_response(
            jsonrpc_error("Rate limit exceeded", None),
            status=429,
        )

    if app_state.api_key:
        token = request.headers.get("x-api-key")
        if token != app_state.api_key:
            return web.json_response(
                jsonrpc_error("Unauthorized", None),
                status=401,
            )

    body = await request.json()
    method = body.get("method")
    params = body.get("params", {})
    request_id = body.get("id")

    if method not in dispatch:
        return web.json_response(jsonrpc_error(f"Unknown method: {method}", request_id))

    try:
        validate_payload(method, params, app_state.schemas)
        result = dispatch[method](params)
        return web.json_response(jsonrpc_response(result, request_id))
    except SchemaValidationError as exc:
        return web.json_response(jsonrpc_error(f"Schema validation failed: {exc}", request_id), status=400)
    except Exception as exc:  # pragma: no cover
        LOGGER.exception("Tool execution failed")
        return web.json_response(jsonrpc_error(str(exc), request_id))


async def run_stdio_server(state: ServerState) -> None:
    dispatch = build_dispatch(state)
    LOGGER.info("Starting stdio JSON-RPC loop")

    while True:
        line = await asyncio.to_thread(input)
        if not line:
            continue

        request = json.loads(line)
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        if method == "shutdown":
            print(json.dumps(jsonrpc_response({"status": "bye"}, request_id)), flush=True)
            return

        if method not in dispatch:
            print(json.dumps(jsonrpc_error(f"Unknown method: {method}", request_id)), flush=True)
            continue

        try:
            validate_payload(method, params, state.schemas)
            result = dispatch[method](params)
            print(json.dumps(jsonrpc_response(result, request_id)), flush=True)
        except SchemaValidationError as exc:
            print(json.dumps(jsonrpc_error(f"Schema validation failed: {exc}", request_id)), flush=True)
        except Exception as exc:
            print(json.dumps(jsonrpc_error(str(exc), request_id)), flush=True)


def validate_registry_contract(registry: dict[str, Any], strict: bool = True) -> dict[str, Any]:
    models = registry.get("models")
    if not isinstance(models, dict):
        raise ValueError("registry.json must define a 'models' object")

    for model_name, entry in models.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{model_name} registry entry must be an object")

        missing = REQUIRED_REGISTRY_FIELDS - entry.keys()
        if missing:
            raise ValueError(f"{model_name} registry entry missing fields: {sorted(missing)}")

        if strict and str(entry.get("sha256", "")).startswith("PENDING_"):
            raise RuntimeError(f"{model_name} has unresolved placeholder hash in strict mode")

    return models


def resolve_model_path(registry_path: str, model_path: str) -> str:
    path = Path(model_path)
    if path.is_absolute():
        return str(path)

    registry_file = Path(registry_path)
    candidate_from_root = registry_file.parent.parent / path
    candidate_from_models = registry_file.parent / path
    if candidate_from_root.exists() or not candidate_from_models.exists():
        return str(candidate_from_root)
    return str(candidate_from_models)


def create_state(config: dict[str, Any]) -> ServerState:
    registry_path = config["models"]["registry_path"]
    strict_startup = bool(config["models"].get("strict_startup", True))
    validate_registry(registry_path, strict=strict_startup)

    detector_cfg = config["detection"]
    perturb_cfg = config["perturbation"]
    alpha = float(config["pipeline"]["alpha_initial"])
    schemas = load_schemas("mcp_server/schemas")

    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    models = validate_registry_contract(registry, strict=strict_startup)
    atn_model_key = str(perturb_cfg.get("model_key", "reface_atn"))
    feedback_model_key = str(config.get("feedback", {}).get("model_key", "deepsafe"))

    atn_entry = models.get(atn_model_key, {})
    feedback_entry = models.get(feedback_model_key, {})

    atn_model_path = resolve_model_path(
        registry_path,
        str(atn_entry.get("path", f"models/{atn_model_key}.pth")),
    )
    feedback_model_path = resolve_model_path(
        registry_path,
        str(feedback_entry.get("path", f"models/{feedback_model_key}.pth")),
    )

    detector = FaceDetector(
        FaceDetectorConfig(
            min_detection_confidence=float(detector_cfg["min_detection_confidence"]),
            min_tracking_confidence=float(detector_cfg["min_tracking_confidence"]),
        )
    )
    atn_engine = ATNEngine(
        ATNConfig(
            model_path=atn_model_path,
            input_size=int(perturb_cfg["atn_input_size"]),
            device=str(perturb_cfg["device"]),
            strict_startup=strict_startup,
            allow_identity_fallback=bool(config["models"].get("allow_identity_fallback", False)),
        )
    )
    blender = FrameBlender(BlendConfig(alpha=alpha))
    init_deepsafe(config, feedback_model_path)

    auth_cfg = config.get("transport", {}).get("auth", {})
    rate_cfg = config.get("transport", {}).get("rate_limit", {})
    limiter = None
    if bool(rate_cfg.get("enabled", False)):
        limiter = SlidingWindowRateLimiter(
            RateLimitConfig(
                requests_per_window=int(rate_cfg.get("requests_per_window", 60)),
                window_seconds=int(rate_cfg.get("window_seconds", 60)),
            )
        )

    return ServerState(
        detector=detector,
        atn_engine=atn_engine,
        blender=blender,
        schemas=schemas,
        api_key=auth_cfg.get("api_key"),
        limiter=limiter,
    )


async def run_http_server(config: dict[str, Any], state: ServerState) -> None:
    cors_cfg = config.get("transport", {}).get("cors", {})
    allowed_origins = cors_cfg.get(
        "allowed_origins",
        ["http://127.0.0.1:5173", "http://localhost:5173"],
    )
    max_payload_mb = int(config.get("transport", {}).get("max_payload_mb", 16))

    @web.middleware
    async def cors_middleware(request: web.Request, handler: Callable[[web.Request], Any]) -> web.StreamResponse:
        origin = request.headers.get("Origin")
        allow_origin = origin if origin in allowed_origins else None

        if request.method == "OPTIONS":
            response = web.Response(status=204)
        else:
            response = await handler(request)

        if allow_origin:
            response.headers["Access-Control-Allow-Origin"] = allow_origin
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type,x-api-key"
            response.headers["Access-Control-Max-Age"] = "86400"
            response.headers["Vary"] = "Origin"

        return response

    app = web.Application(
        middlewares=[cors_middleware],
        client_max_size=max_payload_mb * 1024 * 1024,
    )
    app["state"] = state
    app.router.add_post("/rpc", handle_jsonrpc)
    app.router.add_get("/health", lambda _r: web.json_response({"status": "ok"}))

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=config["transport"]["host"], port=int(config["transport"]["http_port"]))
    await site.start()
    LOGGER.info("HTTP server listening on %s:%s", config["transport"]["host"], config["transport"]["http_port"])

    stop_event = asyncio.Event()
    await stop_event.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adversarial Face Shield MCP server")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    parser.add_argument("--transport", choices=["stdio", "http"], default=None)
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config["logging"]["level"], config["logging"]["output"])

    state = create_state(config)
    transport = args.transport or config["transport"]["mode"]

    if transport == "http":
        await run_http_server(config, state)
    else:
        await run_stdio_server(state)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
