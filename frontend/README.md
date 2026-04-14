# AFS Frontend Console

Production-oriented React + Tailwind dashboard for the Adversarial Face Shield MCP pipeline.

## Capabilities

- Health monitoring against MCP server health endpoint.
- Typed JSON-RPC client for:
  - `face_detector`
  - `perturbation_generator`
  - `frame_blender`
  - `deepfake_feedback`
- Manual frame execution flow with robust request retry and timeout behavior.
- Stream mode (interval-based repeated processing) with start/stop controls.
- Stage-level metrics and runtime event logs.
- Error boundary and user-friendly failure messages.

## Project Structure

```text
src/
  components/
    common/
    dashboard/
  hooks/
  pages/
  services/
  types/
  utils/
```

## Environment Variables

Copy `.env.example` to `.env` and adjust for your environment.

```bash
VITE_AFS_BASE_URL=http://127.0.0.1:8080
VITE_AFS_RPC_ENDPOINT=http://127.0.0.1:8080/rpc
VITE_AFS_HEALTH_ENDPOINT=http://127.0.0.1:8080/health
VITE_AFS_API_KEY=
VITE_AFS_REQUEST_TIMEOUT_MS=20000
VITE_AFS_MAX_RETRIES=2
```

## Local Development

```bash
npm install
npm run dev
```

## Production Build

```bash
npm run build
npm run preview
```

## Integration Notes

- Current browser flow uploads an image and runs the MCP pipeline per frame.
- Stream mode currently re-processes the selected frame on a configurable interval.
- TODO placeholders are included in the UI for:
  - Browser-safe live capture integration (WebRTC or dedicated frame endpoint).
  - Metrics endpoint integration for richer visualizations.
