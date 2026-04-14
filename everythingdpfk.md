# EverythingDPFK

## Scope

This document records implementation and validation details that were directly verified in the latest hardening cycle.

It focuses on:

- Runtime behavior in `src/`, `mcp_server/`, `mcp_client/`, and `frontend/`
- Frontend-backend contract fidelity (JSON-RPC payload/response shape)
- Startup safety, fallback behavior, and observability
- Smoke/benchmark artifacts and model registry integrity

It intentionally avoids speculative roadmap content and keeps claims tied to code edits or command execution.

## Hardening Updates (This Cycle)

### ATN checkpoint compatibility and startup stability

Primary files:

- `src/perturbation/atn_engine.py`
- `scripts/verify_checkpoint.py`

Changes:

- Added filtered checkpoint loading by exact key-and-shape compatibility.
- Added warning logs for skipped checkpoint tensors (missing key / shape mismatch).
- Added identity fallback mode state and `is_identity_mode()` inspection.
- Added architecture fingerprint support:
  - `ReFaceATN.arch_signature()`
  - `compare_arch_signatures(...)`
- Preserved backward compatibility for older checkpoints without `arch` metadata.
- Added `scripts/verify_checkpoint.py` to validate compatibility without starting the server.

Outcome:

- Smoke startup remains stable under partial checkpoint compatibility.
- Strict startup semantics remain fail-fast.
- Fallback decisions are explicit in logs.

### MediaPipe detector fallback hardening

Primary files:

- `src/detection/face_detector.py`
- `scripts/preflight_check.py`
- `requirements.txt`

Changes:

- Added explicit MediaPipe availability check and API-surface handling.
- Added one-time warning when falling back to full-frame detection.
- Included installed MediaPipe version in fallback diagnostics.
- Added `check_mediapipe()` for preflight use.
- Added degraded-state visibility in preflight output.
- Pinned `mediapipe==0.10.9` in requirements for stable `solutions` API behavior.

Outcome:

- Detector degradation is visible and non-silent.
- Fallback path is non-crashing and schema-valid.

### Frontend payload contract and stream determinism fixes

Primary files:

- `frontend/src/services/pipelineService.ts`
- `frontend/src/hooks/usePipelineRunner.ts`

Changes:

- Added exported crop helper (`cropFaceRegion(...)`) for browser-side ROI extraction.
- Added crop padding and image-bound clamping.
- Added `OffscreenCanvas` path with regular canvas fallback.
- Sent JPEG crop payload (quality 0.92) to `perturbation_generator`.
- Preserved full frame payload for `frame_blender`.
- Added perturbation skip behavior on no detection with debug logging.
- Added in-flight guard and skipped-frame counter for interval stream mode.

Outcome:

- Frontend perturbation payload aligns with backend face ROI schema intent.
- Interval stream mode avoids overlapping async executions.

### Smoke runner reliability and config drift fixes

Primary file:

- `scripts/smoke_test_http_pipeline.py`

Changes:

- Added config resolution precedence: CLI > config YAML > environment.
- Added API key masking in logs.
- Removed unsafe hardcoded endpoint fallback behavior.
- Improved startup-failure reporting path and non-blocking stderr handling.

Outcome:

- Smoke runs are profile-consistent and safer to operate.

### Registry normalization and validation

Primary files:

- `scripts/update_registry_hashes.py`
- `scripts/preflight_check.py`
- `src/utils/model_registry.py`
- `mcp_server/server.py`

Changes:

- Refactored registry updater into reusable functions.
- Enforced canonical fields per model entry:
  - `name`
  - `path`
  - `sha256`
  - `architecture`
  - `exported_at`
- Added `validate_registry(...)` in updater script.
- Added registry schema validation to preflight.
- Hardened runtime registry consumption with defensive defaults (`.get(...)`).

Outcome:

- Registry schema is normalized and hash update path is deterministic.
- Preflight explicitly reports schema issues.

### Feedback backend split (runtime default + optional UFD adapter)

Primary files:

- `src/feedback/deepsafe_engine.py`
- `mcp_server/tools/deepfake_feedback.py`
- `config/default.yaml`
- `config/smoke.yaml`
- `config/edge.yaml`

Changes:

- Added shared abstraction: `BaseFeedbackEngine`.
- Added default lightweight runtime backend: `LightweightDeepSafeEngine`.
- Added optional UFD adapter: `UFDDeepSafeAdapter`.
- Added backend factory: `build_feedback_engine(...)`.
- Added `use_ufd_backend: false` to all profiles.

Outcome:

- Runtime remains stable by default.
- UFD parity path is available as an explicit opt-in.

### Test infrastructure and E2E scaffolding

Primary files:

- `tests/conftest.py`
- `pytest.ini`
- `playwright.config.ts`
- `tests/e2e/pipeline_happy_path.spec.ts`
- `tests/e2e/pipeline_error_handling.spec.ts`
- `frontend/package.json`

Changes:

- Added collection-time python path setup (`repo` and `src`).
- Added structural `pythonpath = . src` in `pytest.ini`.
- Added Playwright config and two route-interception E2E specs.
- Added `test:e2e` script in frontend package.

Outcome:

- Unit and integration tests run cleanly.
- Browser E2E coverage is scaffolded and ready.

## Frontend Online Wiring Fix

Observed behavior:

- Frontend showed offline while smoke backend was healthy.

Root cause:

- Frontend defaults pointed to `127.0.0.1:8080`, while smoke backend uses `127.0.0.1:18080`.

Applied fix:

- Added `frontend/.env.local` with:
  - `VITE_AFS_RPC_ENDPOINT=http://127.0.0.1:18080/rpc`
  - `VITE_AFS_HEALTH_ENDPOINT=http://127.0.0.1:18080/health`
  - `VITE_AFS_API_KEY=afs-local-dev-key`
- Fully stopped and restarted Vite dev server after creating/editing env overrides.
- Performed browser hard refresh (`Ctrl+Shift+R`) after restart.

Critical operational rule:

- Vite does not hot-reload `frontend/.env.local` changes at runtime.
- If `.env.local` is created or edited after `npm run dev` starts, the running process keeps stale values.
- Required sequence:
  - stop `npm run dev` completely (`Ctrl+C`)
  - restart from `frontend/` (`npm run dev -- --host 127.0.0.1 --port 5173`)
  - hard refresh browser (`Ctrl+Shift+R`)

Outcome:

- Frontend health polling aligns with smoke profile backend endpoint.
- Frontend offline state caused by stale env read is resolved by restart + hard refresh.

## Latest Local Bring-Up Execution (2026-04-13)

This section records the exact operator flow used to bring the smoke-profile HTTP pipeline online in this workspace, including one startup correction and final verification outcomes.

### Environment and config alignment

Validated frontend override file:

- `frontend/.env.local`

Validated keys:

- `VITE_AFS_RPC_ENDPOINT=http://127.0.0.1:18080/rpc`
- `VITE_AFS_HEALTH_ENDPOINT=http://127.0.0.1:18080/health`
- `VITE_AFS_API_KEY=afs-local-dev-key`

Backend auth key alignment was confirmed against smoke profile config:

- `config/smoke.yaml` -> `transport.auth.api_key: afs-local-dev-key`

### Commands executed

Backend launch (first attempt):

- `d:/deepfake_detection/.venv/Scripts/python.exe -m mcp_server.server --config smoke.yaml --transport http`

Frontend launch:

- `Set-Location frontend`
- `npm run dev -- --host 127.0.0.1 --port 5173`

Verification command:

- `d:/deepfake_detection/.venv/Scripts/python.exe -c "import requests; print('health', requests.get('http://127.0.0.1:18080/health', timeout=5).status_code); print('ui', requests.get('http://127.0.0.1:5173/', timeout=5).status_code)"`

Optional full RPC chain command:

- `d:/deepfake_detection/.venv/Scripts/python.exe -c "import base64, cv2, numpy as np, requests; e='http://127.0.0.1:18080/rpc'; h={'x-api-key':'afs-local-dev-key'}; f=np.full((480,640,3),127,np.uint8); cv2.rectangle(f,(220,120),(420,320),(180,160,130),-1); _,b=cv2.imencode('.png',f); fb=base64.b64encode(b.tobytes()).decode(); call=lambda i,m,p: requests.post(e,headers=h,json={'jsonrpc':'2.0','id':i,'method':m,'params':p},timeout=30).json(); d=call(1,'face_detector',{'frame_b64':fb}); box=(d.get('result') or {}).get('boxes',[[220,120,420,320]])[0]; x1,y1,x2,y2=map(int,box); _,cb=cv2.imencode('.png',f[y1:y2,x1:x2]); pb=base64.b64encode(cb.tobytes()).decode(); p=call(2,'perturbation_generator',{'face_b64':pb}); bl=call(3,'frame_blender',{'frame_b64':fb,'perturbation_b64':p['result']['perturbation_b64'],'boxes':[box],'alpha':0.12}); r=call(4,'deepfake_feedback',{'frame_b64':bl['result']['shielded_frame_b64']}); print('OK', r)"`

### Startup failure, fallback behavior, and correction

Observed first backend failure:

- `Config file not found: smoke.yaml`

Root cause:

- Relative config path was resolved from repository root, while the smoke profile file is located at `config/smoke.yaml`.

Correction applied:

- Relaunched backend with:
  - `d:/deepfake_detection/.venv/Scripts/python.exe -m mcp_server.server --config config/smoke.yaml --transport http`

Post-correction backend state:

- HTTP listener came up on `127.0.0.1:18080`.
- ATN checkpoint compatibility warnings were emitted (non-fatal) and startup continued under the existing relaxed compatibility behavior.
- Lightweight feedback backend was selected as expected by smoke-profile defaults.

### Verification results

Endpoint availability check returned expected values:

- `health 200`
- `ui 200`

Optional JSON-RPC chain result:

- `OK {'jsonrpc': '2.0', 'id': 4, 'result': {'confidence': 0.5358228045944943, 'label': 'fake'}}`

Interpretation:

- Frontend and backend are both online and reachable on the configured local endpoints.
- RPC contract is operational end-to-end across detector, perturbation, blender, and feedback stages.

### Runtime performance notes (this run)

Observed bring-up/runtime indicators:

- Vite dev server became ready in ~`210 ms`.
- Health/UI checks completed within the configured `5s` request timeout window.
- End-to-end RPC quick test completed within the configured `30s` per-call timeout budget and returned a valid result payload.

Operational note:

- If `frontend/.env.local` was created or changed, always restart Vite before hard refresh; refresh alone is not sufficient.

## Execution Evidence

### Core checks executed

- `pytest tests/unit -q` -> pass (`6 passed`)
- `pytest tests/integration -q` -> pass (`1 passed`)
- Frontend lint/build -> pass
- Backend `/health` on smoke endpoint -> pass
- Full JSON-RPC chain execution -> pass:
  - `face_detector`
  - `perturbation_generator`
  - `frame_blender`
  - `deepfake_feedback`

### Smoke artifact

From `validation/smoke_result.json`:

- `status`: `ok`
- `endpoint`: `http://127.0.0.1:18080/rpc`
- `tools_called`: all 4 expected tools
- `face_detector.error`: `mediapipe_unavailable_full_frame_fallback`

Interpretation:

- End-to-end tool chain is operational.
- Detector is running in explicit degraded fallback mode in this environment.

### Benchmark artifact

From `validation/benchmark_results.json`:

- `frames`: `40`
- `latency_ms.p95`: `915.5403999975533`
- `latency_ms.mean`: `887.8321949995552`
- `ssim.mean`: `0.990357188345845`

Interpretation:

- CPU local benchmark remains detector-dominant.
- Results are consistent with smoke-profile expectations.

### Registry integrity

From `models/registry.json`:

- `reface_atn.sha256`: `d0cf0861bd9da0e982f8c8c167878338180700033d02199cd046d1d1bb6d13b6`
- `deepsafe.sha256`: `6ef9d4ba27ac325f77a2eb678d511d3440ff08045cc661da83fd5a183c357eee`

## Fallbacks and Corrections (Operational Summary)

Fallbacks now explicit and observable:

- ATN identity fallback mode (relaxed startup only).
- Full-frame detection fallback when MediaPipe mesh API is unavailable.
- Lightweight feedback backend as default, with optional UFD adapter.

Corrections implemented for stability and determinism:

- Frontend sends ROI crop for perturbation stage.
- Stream interval avoids overlapping async runs.
- Smoke script resolves endpoint/key deterministically and masks secrets.
- Preflight catches registry schema issues and detector degradation early.

## Known Divergences and Practical Notes

- Runtime default feedback backend may differ from notebook detector backend unless `use_ufd_backend` is enabled.
- MediaPipe API availability remains environment-dependent; fallback mode is expected in some builds.
- ATN checkpoint compatibility can be partial for older checkpoints; use `scripts/verify_checkpoint.py` before enabling strict startup in production.

## Recommended Run Order

1. Run preflight and inspect degraded-state warnings.
2. Run registry hash update and schema validation.
3. Start backend with selected profile.
4. Ensure frontend env points to the same profile endpoint.
5. Run smoke chain and verify artifact output.
6. Run benchmark for latest latency/quality snapshot.

## Confidence Statement

This document reflects only what was observed in code and terminal execution during this hardening cycle. It should be updated whenever detector backend selection, ATN architecture compatibility behavior, frontend endpoint wiring, or registry schema rules change.
