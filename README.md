# Adversarial Face Shield (AFS)

Production-oriented reference implementation of a modular, JSON-RPC-driven pipeline that perturbs facial regions to reduce downstream deepfake/face-recognition utility.

## What Is Implemented Right Now

- MCP-style JSON-RPC server with four callable tools:
  - `face_detector`
  - `perturbation_generator`
  - `frame_blender`
  - `deepfake_feedback`
- Browser frontend that exercises the toolchain end-to-end over HTTP.
- Capture client for webcam loop + optional virtual camera output.
- Adaptive alpha controller driven by feedback confidence.
- Schema validation for all tool payloads (JSON Schema).
- API key auth + sliding-window rate limiting on the HTTP transport.
- Model registry integrity checks with SHA256 validation.
- Smoke + benchmark scripts and unit/integration/E2E scaffolding.

## Runtime Architecture

1. Input frame arrives from browser or capture client.
2. `face_detector` returns boxes + landmarks (or explicit fallback error mode).
3. `perturbation_generator` receives a face ROI crop and returns perturbation image bytes.
4. `frame_blender` merges perturbation into each ROI using alpha + smoothing blur.
5. Optional `deepfake_feedback` returns confidence and label, used to update alpha.

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
- HTTP transport:
  - Supports CORS allow-list and configurable max payload size.
  - Uses API key header `x-api-key` when configured.

## Project Layout (Primary Paths)

- `mcp_server/`: JSON-RPC server, auth/rate limiting, schema validation, tool handlers.
- `mcp_client/`: capture loop, alpha controller, virtual camera driver.
- `src/detection/`: MediaPipe-based face detector + fallback behavior.
- `src/perturbation/`: ATN model, compatibility-aware checkpoint loading.
- `src/blending/`: ROI blending and output clipping.
- `src/feedback/`: feedback backends (lightweight + optional UFD adapter).
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

## Running the Stack

### 1) Preflight

```powershell
python scripts/preflight_check.py
```

### 2) Start HTTP Backend (Smoke Profile)

```powershell
python -m mcp_server.server --config config/smoke.yaml --transport http
```

### 3) Start Frontend

```powershell
Set-Location frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

### 4) Optional Capture Client

```powershell
python -m mcp_client.capture_client --config config/default.yaml --rpc-endpoint http://127.0.0.1:8080/rpc --virtual-cam
```

Press `q` in the OpenCV window to stop the capture loop.

## Testing and Validation

### Python Tests

```powershell
pytest tests/unit -q
pytest tests/integration -q
```

### Frontend E2E (Playwright)

```powershell
Set-Location frontend
npm run test:e2e
```

### Smoke Script

```powershell
python scripts/smoke_test_http_pipeline.py --python .venv/Scripts/python.exe --config config/smoke.yaml
```

Artifacts:
- `validation/smoke_result.json`
- `validation/smoke_output.png`

### Local Benchmark Script

```powershell
python scripts/run_phase5_benchmark_local.py --config config/smoke.yaml --frames 40
```

Artifact:
- `validation/benchmark_results.json`

## Model Registry Maintenance

Update model hashes:

```powershell
python scripts/update_registry_hashes.py
```

Verify ATN checkpoint compatibility directly:

```powershell
python scripts/verify_checkpoint.py --checkpoint models/reface_atn.pth --device cpu
```

## Docker

```powershell
docker compose up --build
```

This starts:
- `mcp-server` (HTTP)
- `capture-client`

## Data and Model Reference

For current inventory, paths, and provenance details:
- `DATA_AND_MODELS.md`

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
