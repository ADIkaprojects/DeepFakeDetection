# Adversarial Face Shield (AFS)

Production-oriented reference implementation of a modular, JSON-RPC-driven pipeline that perturbs facial regions to reduce downstream deepfake/face-recognition utility.

## What Is Implemented Right Now

- MCP-style JSON-RPC server with five callable tools:
  - `face_detector`
  - `perturbation_generator`
  - `frame_blender`
  - `deepfake_feedback`
  - `nsfw_feedback`
- Browser frontend that exercises the toolchain end-to-end over HTTP.
- Capture client for webcam loop + optional virtual camera output.
- Adaptive alpha controller driven by feedback confidence.
- Schema validation for all tool payloads (JSON Schema).
- API key auth + sliding-window rate limiting on the HTTP transport.
- Model registry integrity checks with SHA256 validation.
- NSFW-trigger perturbation modes integrated into `perturbation_generator`:
  - `shield_only` (default, backward-compatible)
  - `nsfw_trigger_only`
  - `shield_and_nsfw`
- Smoke + benchmark scripts and unit/integration/E2E scaffolding.

## Runtime Architecture

1. Input frame arrives from browser or capture client.
2. `face_detector` returns boxes + landmarks (or explicit fallback error mode).
3. `perturbation_generator` receives a face ROI crop and returns perturbation image bytes.
   - Default route uses existing shield path.
   - Optional route uses dual-head perturbation path selected by `protection_profile`.
4. `frame_blender` merges perturbation into each ROI using alpha + smoothing blur.
5. Optional `deepfake_feedback` returns confidence and label, used to update alpha.
6. Optional `nsfw_feedback` scores the processed frame and returns NSFW label/score.

### NSFW-Trigger Path (Implementation)

- `src/perturbation/nsfw_trigger_atn.py`
  - Added `NSFWTriggerATN` model (encoder/decoder + residual bottleneck).
  - Added bounded delta generation (`l_inf_bound`) and clamp to preserve imperceptibility bounds.
  - Added output-size correction in `forward()` to keep decoder output exactly aligned with input ROI size.
- `src/perturbation/atn_engine.py`
  - Added `DualHeadATNEngine` to combine existing shield perturbation with optional NSFW-trigger perturbation.
  - Added profile dispatch logic for `shield_only`, `nsfw_trigger_only`, and `shield_and_nsfw`.
- `src/perturbation/perturbation_combiner.py`
  - Added weighted combination, joint L-infinity cap, and SSIM-proxy guard.
- `src/feedback/nsfw_feedback_engine.py`
  - Added proxy-based NSFW scorer (`NSFWProxyEnsemble`) using HuggingFace image classification runtime.
  - Added runtime output contract with `nsfw_score`, `label`, and `proxies_used`.
- `mcp_server/tools/nsfw_feedback.py`
  - Added new MCP tool handler for NSFW feedback.
- `mcp_server/schemas/nsfw_feedback.schema.json`
  - Added schema for `nsfw_feedback` request payload.
- `mcp_server/schemas/perturbation_generator.schema.json`
  - Extended perturbation request schema with optional `protection_profile` enum.

## Known Runtime Behaviors

- MediaPipe fallback:
  - If FaceMesh API is unavailable, detector returns full-frame fallback with:
    - `error = mediapipe_unavailable_full_frame_fallback`
- ATN checkpoint loading:
  - In strict mode, incompatible checkpoint fields fail startup.
  - In relaxed mode, compatible tensors are partially loaded.
  - Optional identity fallback can run when checkpoint loading cannot proceed.
- Feedback backend:
  - Default runtime backend is lightweight and deterministic.
  - Optional UFD adapter can be enabled for closer parity with heavy model path.
- NSFW feedback behavior:
  - Scores processed frames using configured NSFW proxies.
  - Returns explicit error payload on invalid input.
  - Uses threshold `0.5` for `safe` vs `nsfw_flagged` label.
- HTTP transport:
  - Supports CORS allow-list and configurable max payload size.
  - Uses API key header `x-api-key` when configured.

## Project Layout (Primary Paths)

- `mcp_server/`: JSON-RPC server, auth/rate limiting, schema validation, tool handlers.
- `mcp_client/`: capture loop, alpha controller, virtual camera driver.
- `src/detection/`: MediaPipe-based face detector + fallback behavior.
- `src/perturbation/`: ATN models, compatibility-aware checkpoint loading, NSFW trigger and perturbation combiner.
- `src/blending/`: ROI blending and output clipping.
- `src/feedback/`: deepfake feedback backends (lightweight + optional UFD adapter) plus NSFW proxy feedback engine.
- `frontend/`: React/Vite dashboard + pipeline controls and health polling.
- `scripts/`: preflight, smoke test, benchmark, registry hash updates.
- `config/`: `default.yaml`, `smoke.yaml`, `edge.yaml` profiles.
- `models/registry.json`: canonical model metadata + hashes.

## Prerequisites

- Python 3.11 recommended.
- Optional CUDA-capable GPU for faster perturbation inference.
- Node.js 20+ for frontend development.
- Model files present:
  - `models/reface_atn.pth`
  - `models/deepsafe.pth`
  - `models/nsfw_trigger_atn.pth` (required for trained NSFW-trigger profile behavior)

## Setup

### Python Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Frontend Environment

```powershell
Set-Location frontend
npm install
```

Recommended local frontend overrides (`frontend/.env.local`):

```env
VITE_AFS_RPC_ENDPOINT=http://127.0.0.1:18080/rpc
VITE_AFS_HEALTH_ENDPOINT=http://127.0.0.1:18080/health
VITE_AFS_API_KEY=afs-local-dev-key
```

Important: after changing `frontend/.env.local`, fully restart Vite.

### NSFW Proxy Provisioning

Download and cache NSFW proxy model artifacts used by `nsfw_feedback`:

```powershell
.\.venv\Scripts\python.exe scripts/download_nsfw_proxies.py
```

Notes:

- Artifacts are placed under `models/nsfw_proxy/`.
- These files are large and should be excluded from normal git commits unless Git LFS is configured.

## Running the Stack

### 1) Preflight

```powershell
.\.venv\Scripts\python.exe scripts/preflight_check.py
```

### 2) Start HTTP Backend (Smoke Profile)

```powershell
.\.venv\Scripts\python.exe -m mcp_server.server --config config/smoke.yaml --transport http
```

### 3) Start Frontend

```powershell
Set-Location frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

### 4) Optional Capture Client

```powershell
.\.venv\Scripts\python.exe -m mcp_client.capture_client --config config/default.yaml --rpc-endpoint http://127.0.0.1:8080/rpc --virtual-cam
```

Press `q` in the OpenCV window to stop the capture loop.

## Testing and Validation

### Python Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit -q
.\.venv\Scripts\python.exe -m pytest tests/integration -q
```

NSFW-focused checks:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_nsfw_trigger_atn.py -q
.\.venv\Scripts\python.exe -m pytest tests/integration/test_nsfw_feedback.py -q
```

### Frontend E2E (Playwright)

```powershell
Set-Location frontend
npm run test:e2e
```

### Smoke Script

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test_http_pipeline.py --python .venv/Scripts/python.exe --config config/smoke.yaml --api-key afs-local-dev-key
```

Artifacts:
- `validation/smoke_result.json`
- `validation/smoke_output.png`

Latest verified NSFW-inclusive smoke output:

- `status = ok`
- `tools_called` includes `nsfw_feedback` and `perturbation_generator(shield_and_nsfw)`
- `deepfake_feedback.confidence = 0.5360739988221854`
- `nsfw_feedback.nsfw_score = 0.012855`
- `nsfw_feedback.label = safe`
- `nsfw_profile = shield_and_nsfw`

### Local Benchmark Script

```powershell
.\.venv\Scripts\python.exe scripts/run_phase5_benchmark_local.py --config config/smoke.yaml --frames 40
```

Artifact:
- `validation/benchmark_results.json`

## Model Registry Maintenance

Update model hashes:

```powershell
.\.venv\Scripts\python.exe scripts/update_registry_hashes.py
```

Verify ATN checkpoint compatibility directly:

```powershell
.\.venv\Scripts\python.exe scripts/verify_checkpoint.py --checkpoint models/reface_atn.pth --device cpu
```

Validate NSFW trigger checkpoint behavior:

```powershell
.\.venv\Scripts\python.exe scripts/validate_nsfw_trigger.py --image <path-to-face-image> --checkpoint models/nsfw_trigger_atn.pth --device cpu
```

Current validation thresholds:

- `nsfw_score_after > 0.5`
- `SSIM > 0.97`
- `L_inf <= 0.06`

## Docker

```powershell
docker compose up --build
```

This starts:
- `mcp-server` (HTTP)
- `capture-client`

## Data and Model Reference

For current inventory, paths, and provenance details:
- `validation/DATA_AND_MODELS.md`

## Fallbacks, Fixes, and Corrections Applied

### 1) Combined profile ROI shape mismatch

Issue observed during smoke path execution:

- Combined perturbation failed on some odd ROI dimensions due to width mismatch in composed tensors.

Correction:

- Added explicit interpolation in `NSFWTriggerATN.forward()` when decoder output size differs from input size.

Outcome:

- `shield_and_nsfw` profile runs successfully in smoke validation.

### 2) HuggingFace preprocessing mismatch

Issue observed:

- Download and inference path initially used mismatched preprocessing APIs.

Correction:

- Standardized NSFW proxy loader and scorer on `AutoImageProcessor`.

Outcome:

- Consistent preprocessing between downloaded proxy assets and runtime scoring.

### 3) Smoke execution hardening

Corrections:

- Deterministic endpoint/key resolution order: CLI > config > environment.
- API-key masking in logs.
- Startup/health probing and process lifecycle handling made more robust.

Outcome:

- Stable and repeatable smoke runs.

### 4) Backward compatibility preservation

Preserved behavior:

- Existing shield-only flow remains default and unchanged.
- New NSFW behavior is opt-in through `protection_profile` and/or `nsfw_feedback` calls.

## Starting GitHub Deployment (Initial Bootstrap)

This workspace was not initialized as a git repository at the time of this update.

Recommended bootstrap sequence:

1. Initialize local git repo.
2. Add a root `.gitignore` to exclude heavy/generated/private artifacts.
3. Commit baseline source and docs.
4. Create an empty GitHub repository.
5. Add remote and push.

Commands:

```powershell
git init
git add .
git commit -m "Initial commit: AFS pipeline"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

If pushing fails due to large files, track only source/docs and move large model/data artifacts to release storage or Git LFS.

## Contributing

### Branching

- Create feature branches from `main`:
  - `feature/<short-name>`
- Keep PRs focused and small when possible.

### Local Validation Before PR

Run the minimum quality gate:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit -q
.\.venv\Scripts\python.exe -m pytest tests/integration -q
```

For frontend changes, also run:

```powershell
Set-Location frontend
npm run lint
npm run build
npm run test:e2e
```

### Commit Guidelines

- Use clear, action-oriented commit messages.
- Mention impacted subsystem when useful (for example: `server`, `frontend`, `detection`, `scripts`).
- Update docs/config examples when behavior changes.

## Release Checklist

Use this checklist before tagging or sharing a release build.

1. Verify config profile to be shipped (`config/default.yaml`, `config/smoke.yaml`, or `config/edge.yaml`).
2. Run preflight and resolve all blockers.
3. Validate model registry integrity and hashes.
4. Execute smoke pipeline and confirm artifact output.
5. If NSFW path is included, run NSFW unit/integration checks and confirm `nsfw_feedback` in smoke output.
6. Run Python unit/integration tests.
7. Run frontend lint/build and E2E tests if frontend is part of release.
8. Confirm README and operational docs match runtime behavior.
9. Tag release and push tag.

Suggested release commands:

```powershell
git tag -a v0.1.0 -m "AFS v0.1.0"
git push origin v0.1.0
```
