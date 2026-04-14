# Adversarial Face Shield - Execution Roadmap

## 1. Hierarchical Decomposition

### 1.1 Program-Level Goals
- Build a real-time deepfake-resilient face shielding system with end-to-end latency under 60 ms.
- Keep architecture modular and protocol-driven via JSON-RPC/MCP-style tool boundaries.
- Ensure production readiness through validation, observability, and containerization.

### 1.2 System-Level Subsystems
- Frame ingress and egress: webcam capture, virtual camera output.
- Vision preprocessing: face detection, landmark extraction, alignment.
- Perturbation inference: ATN loading, preprocessing, GPU inference, postprocessing.
- Frame synthesis: perturbation blending, artifact smoothing, clipping.
- Adaptive control: feedback scheduling, confidence smoothing, alpha controller.
- Tool runtime: stateless MCP tool adapters and server transport layer.
- Quality controls: model registry validation, test suites, benchmarks, logs/metrics.

### 1.3 Module Breakdown
- `src/detection`: detector abstraction and MediaPipe backend.
- `src/perturbation`: ATN engine and tensor transforms.
- `src/blending`: ROI blending and seam smoothing.
- `src/utils`: config, logging, encoding, timing, schema/model registry checks.
- `mcp_server/tools`: protocol handlers for each pipeline capability.
- `mcp_client`: orchestration loop, alpha control, virtual camera driver.

## 2. Dependency-Aware Build Order
1. Foundation: config, logging, data contracts, model registry validation.
2. Core modules: detection, perturbation, blending.
3. Prototype loop: monolithic script for fast local validation.
4. MCP modularization: server, schemas, tool handlers.
5. Client orchestration: capture loop and adaptive alpha feedback.
6. Integration and benchmark tests.
7. Packaging: Docker, compose, CI.
8. Edge path prep: ONNX export hooks and TensorRT placeholders.

## 3. Phase-Mapped Execution Plan

### Phase 0 - Environment Setup
- Build reproducible Python environment.
- Validate CUDA/device availability.
- Validate model registry and file checksums.

### Phase 1 - Monolithic Prototype
- Implement and wire detection + ATN + blending in `prototype/pipeline.py`.
- Measure latency and SSIM baseline.

### Phase 2 - MCP Modularization
- Implement JSON-RPC server with tool registration and typed payload validation.
- Expose tools: `face_detector`, `perturbation_generator`, `frame_blender`, `deepfake_feedback`.

### Phase 3 - Adaptive Feedback
- Add feedback scheduler and rolling confidence smoothing.
- Implement alpha control bounds, escalation, and decay.

### Phase 4 - Virtual Camera Integration
- Route processed frames to `pyvirtualcam`.
- Add frame pacing and graceful shutdown.

### Phase 5 - Validation and Benchmarks
- Add unit/integration tests and benchmark scaffolds.
- Capture stage timing and produce validation report template.

### Phase 6 - Containerization
- Add Dockerfiles, Compose stack, CI workflow.

### Phase 7 - Edge Optimization Preparation
- Add ONNX export script and TensorRT readiness notes/config.

## 4. Prioritized TODO Backlog

### P0 (Must-Have)
- [ ] Build core pipeline modules with strict type hints and robust error handling.
- [ ] Implement JSON-RPC server and tool handlers.
- [ ] Implement capture client with adaptive alpha and virtual camera output.
- [ ] Add model registry checksum validation at startup.
- [ ] Add structured logging and latency instrumentation.

### P1 (Should-Have)
- [ ] Add end-to-end integration tests and synthetic frame flow tests.
- [ ] Add benchmark script for P50/P95/P99 stage latency.
- [ ] Add health endpoint and readiness checks for server mode.

### P2 (Future / Phase 7+)
- [ ] Add TensorRT engine runtime adapter.
- [ ] Add binary payload transport to reduce Base64 overhead.
- [ ] Add multi-face batching and backpressure queue.

## 5. Verification Strategy per Step
- Run static checks and focused tests after each module group.
- Simulate frame flow with synthetic image fixtures for deterministic validation.
- Confirm runtime initialization and startup stability before next phase.
- Track regressions through latency counters and structured logs.
