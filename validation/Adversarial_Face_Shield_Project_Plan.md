# Adversarial Face Shield — Deepfake Immune System
## Comprehensive Project Plan & Implementation Roadmap

**Document Version:** 1.0  
**Classification:** Internal — Technical & Stakeholder Distribution  
**Prepared By:** AI Project Planning Office  
**Date:** April 2026  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Scope & Objectives](#2-project-scope--objectives)
3. [Tools & Technologies Inventory](#3-tools--technologies-inventory)
4. [System Architecture Overview](#4-system-architecture-overview)
5. [Project Phases & Milestones](#5-project-phases--milestones)
6. [Detailed Workflow by Phase](#6-detailed-workflow-by-phase)
7. [Roles & Responsibilities](#7-roles--responsibilities)
8. [Integration Points & Key Functionality](#8-integration-points--key-functionality)
9. [Risk Register & Mitigation Strategies](#9-risk-register--mitigation-strategies)
10. [Quality Assurance & Validation Framework](#10-quality-assurance--validation-framework)
11. [Deployment Strategy](#11-deployment-strategy)
12. [Appendices](#12-appendices)

---

## 1. Executive Summary

### 1.1 Project Overview

The **Adversarial Face Shield (AFS)** is a real-time, AI-powered privacy protection layer designed to prevent unauthorized facial data harvesting, deepfake generation, and biometric cloning. Operating as the core component of a broader **Deepfake Immune System (DIS)**, the AFS intercepts a user's video stream before transmission, applies imperceptible adversarial perturbations to facial regions, and re-outputs the modified stream to any downstream application — rendering the face effectively unusable for deepfake model training or face-recognition attacks.

The system achieves this through a tightly orchestrated pipeline of computer vision, deep learning inference, and a modular **Model Context Protocol (MCP)** server backbone that decouples each processing stage for independent scalability, testability, and extension.

### 1.2 Business Justification

Deepfake technology has matured to a point where high-fidelity face-swapping is achievable with only a few minutes of source video — footage that is routinely captured in video calls, livestreams, and social media. The Adversarial Face Shield addresses this threat proactively, operating at the source of capture rather than relying on downstream detection alone. This positions the system as both a personal privacy tool and an enterprise-grade trust layer for organizations whose executives, spokespeople, and remote workers face elevated identity-spoofing risks.

### 1.3 Success Criteria

| Criterion | Target Threshold |
|---|---|
| Face-recognition attack success rate (post-shield) | < 20% (baseline ~80%) |
| Structural Similarity Index (SSIM) — visual fidelity | ≥ 0.95 |
| Mean Opinion Score (MOS) — user perception | ≥ 4.0 / 5.0 |
| End-to-end pipeline latency (capture → shield → output) | < 60 ms |
| System uptime / reliability | ≥ 99.5% |
| Virtual camera compatibility | ≥ 5 major conferencing platforms |

---

## 2. Project Scope & Objectives

### 2.1 In-Scope

- Real-time adversarial perturbation of facial regions in video streams
- Face detection and landmark-based alignment (preprocessing)
- MCP server with modular tool architecture managing all pipeline communications
- Optional deepfake detection feedback loop for adaptive perturbation scaling
- Virtual camera output compatible with Zoom, Teams, Google Meet, OBS, and Webex
- Docker-based containerized deployment
- Documentation, testing suite, and CI/CD pipeline
- Edge device deployment target (NVIDIA Jetson Nano)

### 2.2 Out of Scope (Phase 1)

- Mobile application (iOS/Android) integration
- Audio deepfake protection
- Cloud-hosted SaaS offering (deferred to Phase 3)
- End-user GUI application (basic CLI/tray icon only in Phase 1)

### 2.3 Core Objectives

1. **Prototype a functional real-time Adversarial Face Shield** leveraging pretrained Adversarial Transformation Networks (ATNs).
2. **Architect a modular MCP-based server** that cleanly separates detection, perturbation, blending, and feedback concerns.
3. **Validate imperceptibility and effectiveness** using SSIM metrics and commercial face-recognition API attack-success benchmarks.
4. **Deliver containerized deployment artifacts** enabling reproducible installation across diverse environments.

---

## 3. Tools & Technologies Inventory

This section details every technology component in the project, its role, the rationale for selection, and integration notes.

---

### 3.1 Adversarial Transformation Network — ReFace

| Attribute | Detail |
|---|---|
| **Repository** | `ReFace-attack/ReFace` |
| **Type** | Pretrained Feed-Forward U-Net (ATN) |
| **Purpose** | Core perturbation engine — generates adversarial noise that degrades face-recognition embeddings |
| **Key Advantage** | Real-time capable; single forward pass produces perturbation without iterative optimization |
| **Integration Point** | MCP `perturbation_generator` tool; called per-face-ROI per frame |
| **Checkpoint** | `reface_atn.pth` — frozen weights, inference only |

**Usage Notes:** The ATN takes a normalized, aligned face crop as input and outputs a perturbation tensor of identical shape. The perturbation is scaled by an `alpha` factor (default: 0.12) before additive blending. The `alpha` value is the primary adaptive control parameter managed by the feedback loop.

---

### 3.2 ApaNet — Adversarial Noise Alleviation (Optional Baseline)

| Attribute | Detail |
|---|---|
| **Source** | PMC / published architecture (stacked residual blocks) |
| **Purpose** | Baseline comparison — used to evaluate how well the AFS perturbations survive downstream denoising attempts |
| **Integration Point** | Validation & red-team testing only; not in the live pipeline |

---

### 3.3 OpenCV

| Attribute | Detail |
|---|---|
| **Package** | `opencv-python` |
| **Purpose** | Frame capture from webcam (`VideoCapture`), image encode/decode (PNG/JPEG), pixel arithmetic for blending, virtual camera frame formatting |
| **Integration Point** | Capture client (`capture_client.py`); frame blender tool in MCP server |

**Usage Notes:** OpenCV operates at the entry and exit points of the pipeline. It must be compiled with V4L2 support on Linux for virtual camera output. Frame timestamps must be monotonically increasing to prevent virtual camera stutter.

---

### 3.4 dlib / MediaPipe — Face Detection & Landmark Extraction

| Attribute | Detail |
|---|---|
| **Packages** | `dlib`, `mediapipe` |
| **Purpose** | Face bounding-box detection and 68-point landmark extraction for precise ROI cropping and alignment |
| **Primary Choice** | MediaPipe FaceMesh (468 landmarks, GPU-accelerated, ~2 ms latency) |
| **Fallback** | dlib HOG-based detector (CPU-friendly, proven robustness) |
| **Integration Point** | MCP `face_detector` tool |

**Usage Notes:** Landmark data is used not only for cropping but also for affine alignment — normalizing head pose before the ATN forward pass significantly improves perturbation effectiveness. The bounding box coordinates are returned in JSON format and used by the capture client to re-insert perturbed faces accurately.

---

### 3.5 PyTorch / TorchVision

| Attribute | Detail |
|---|---|
| **Packages** | `torch`, `torchvision` |
| **Purpose** | ATN model loading, GPU inference, tensor normalization/denormalization |
| **CUDA Support** | Required for < 30 ms inference on GPU; CPU fallback for edge/testing |
| **Integration Point** | MCP `perturbation_generator` tool |

**Usage Notes:** The ATN checkpoint is loaded once at server startup and kept resident in GPU memory. `torch.no_grad()` context is mandatory during inference to disable gradient tracking and reduce memory overhead.

---

### 3.6 Model Context Protocol (MCP) Server

| Attribute | Detail |
|---|---|
| **Specification** | Model Context Protocol 2025-03-26 |
| **SDK** | `mcp-sdk` (Python reference implementation) or custom JSON-RPC 2.0 over WebSockets |
| **Purpose** | Messaging backbone decoupling all pipeline stages; tool registration, lifecycle management, capability negotiation |
| **Transport Options** | `stdio` (local, lowest latency) or Streamable HTTP (remote/containerized) |
| **Integration Point** | Central hub — all pipeline modules register as tools and communicate exclusively through this server |

**MCP Tools Registered:**

| Tool Name | Input | Output |
|---|---|---|
| `face_detector` | Base64-encoded frame PNG | Bounding boxes + landmarks JSON |
| `perturbation_generator` | Base64-encoded aligned face PNG | Base64-encoded perturbation tensor |
| `frame_blender` | Frame B64 + perturbation B64 + alpha | Base64-encoded shielded frame PNG |
| `deepfake_feedback` *(optional)* | Base64-encoded shielded frame | Deepfake confidence score (0–1) |

---

### 3.7 DeepSafe — Feedback Detection Model

| Attribute | Detail |
|---|---|
| **Repository** | `siddharthksah/DeepSafe` |
| **Purpose** | Adaptive feedback loop — evaluates shielded frame for residual deepfake susceptibility; result drives dynamic `alpha` adjustment |
| **Architecture** | CNN-LSTM multi-model ensemble with Streamlit UI |
| **Integration Point** | MCP `deepfake_feedback` tool (optional, toggled via config flag) |

**Usage Notes:** The feedback tool is invoked every N frames (configurable; default: every 5 frames) rather than per-frame to reduce latency overhead. If returned confidence > 0.70, `alpha` is incremented by 0.02 (capped at 0.30) for subsequent frames.

---

### 3.8 pyvirtualcam / OBS-VirtualCam

| Attribute | Detail |
|---|---|
| **Package** | `pyvirtualcam` |
| **Purpose** | Expose the shielded video stream as a virtual webcam device that is selectable in any conferencing application |
| **Platform Support** | Linux (V4L2), Windows (OBS-VirtualCam), macOS (Continuity Camera workaround) |
| **Integration Point** | Output stage of `capture_client.py` |

---

### 3.9 TensorRT (Edge Deployment)

| Attribute | Detail |
|---|---|
| **Purpose** | Optimize ATN model for NVIDIA Jetson Nano deployment; achieves sub-30 ms inference on edge hardware |
| **Target Hardware** | Jetson Nano 4GB |
| **Integration Point** | Replaces standard PyTorch inference in edge deployment profile |

---

### 3.10 Docker & Kubernetes

| Attribute | Detail |
|---|---|
| **Purpose** | Containerize the MCP server and client components for reproducible, environment-agnostic deployment |
| **Compose Profile** | `docker-compose.yml` defines `mcp-server`, `capture-client`, and optional `feedback-service` containers |
| **Kubernetes Profile** | MCP server as sidecar; capture clients as DaemonSets on workstation nodes |

---

## 4. System Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        ADVERSARIAL FACE SHIELD SYSTEM                      │
│                                                                            │
│  ┌─────────────┐     Raw Frame      ┌────────────────────────────────────┐ │
│  │   Webcam /  │ ─────────────────► │         MCP SERVER                 │ │
│  │  Video Src  │                    │                                    │ │
│  └─────────────┘                    │  ┌──────────────────────────────┐  │ │
│                                     │  │  Tool: face_detector         │  │ │
│  ┌─────────────────────────────┐    │  │  (MediaPipe / dlib)          │  │ │
│  │     CAPTURE CLIENT          │    │  └──────────────┬───────────────┘  │ │
│  │                             │    │                 │ boxes + landmarks │ │
│  │  1. Capture Frame           │    │  ┌──────────────▼───────────────┐  │ │
│  │  2. Call face_detector ────►│────┤  │  Tool: perturbation_generator│  │ │
│  │  3. Crop face ROI           │    │  │  (ReFace ATN — PyTorch/GPU)  │  │ │
│  │  4. Call perturbation_gen ─►│────┤  └──────────────┬───────────────┘  │ │
│  │  5. Call frame_blender ────►│────┤                 │ pert. tensor      │ │
│  │  6. Call deepfake_feedback ►│────┤  ┌──────────────▼───────────────┐  │ │
│  │     (every N frames)        │    │  │  Tool: frame_blender         │  │ │
│  │  7. Adjust alpha            │    │  │  (OpenCV addWeighted)        │  │ │
│  │  8. Push to virtual cam     │    │  └──────────────┬───────────────┘  │ │
│  └─────────────────────────────┘    │                 │ shielded frame    │ │
│                                     │  ┌──────────────▼───────────────┐  │ │
│  ┌─────────────┐                    │  │  Tool: deepfake_feedback      │  │ │
│  │  Virtual    │◄── shielded frame ─┤  │  (DeepSafe CNN-LSTM)         │  │ │
│  │  Webcam     │                    │  └──────────────────────────────┘  │ │
│  │  (pyvirtcam)│                    └────────────────────────────────────┘ │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │         Conferencing App (Zoom / Teams / Meet / OBS / Webex)        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Data Flow Summary

1. The **Capture Client** reads raw frames from the webcam at target FPS (default: 30).
2. Each frame is Base64-encoded and dispatched to the MCP server's `face_detector` tool via JSON-RPC 2.0.
3. Returned bounding boxes are used to crop face ROIs, which are forwarded to `perturbation_generator`.
4. The ATN outputs an adversarial perturbation tensor.
5. Both the original frame and the perturbation are sent to `frame_blender`, which applies additive blending with alpha scaling.
6. Every N frames, the shielded frame is submitted to `deepfake_feedback`; the confidence score drives alpha adjustment.
7. The shielded frame is written to the virtual camera device.

### 4.2 Communication Protocol

All inter-component communication uses **JSON-RPC 2.0** framing over the MCP transport layer. Local deployments use `stdio` for sub-millisecond IPC overhead. Remote or container-isolated deployments use **Streamable HTTP** with persistent connections. All payloads carrying image data use Base64 encoding over the wire; frame compression is PNG (lossless) by default, switchable to JPEG (quality 95) for bandwidth-constrained environments.

---

## 5. Project Phases & Milestones

### Phase Overview

| Phase | Name | Duration | Key Deliverable |
|---|---|---|---|
| **Phase 0** | Foundation & Environment Setup | 1 week | Reproducible dev environment; all repos cloned and validated |
| **Phase 1** | Core Pipeline Prototype | 3 weeks | Functional single-machine AFS prototype (no MCP) |
| **Phase 2** | MCP Server Integration | 2 weeks | Modular MCP-based pipeline with all 4 tools registered |
| **Phase 3** | Feedback Loop & Adaptive Alpha | 1 week | DeepSafe feedback integrated; adaptive alpha control proven |
| **Phase 4** | Virtual Camera & UX Integration | 1 week | pyvirtualcam output; confirmed compatibility with ≥3 conferencing apps |
| **Phase 5** | Validation & Benchmarking | 2 weeks | All success criteria formally measured and documented |
| **Phase 6** | Containerization & Deployment | 1 week | Docker images published; Kubernetes manifests ready |
| **Phase 7** | Edge Optimization (Jetson) | 2 weeks | TensorRT-optimized ATN; < 30 ms latency on Jetson Nano |
| **Phase 8** | Documentation & Handover | 1 week | Full technical documentation; runbooks; stakeholder presentation |

**Total Estimated Duration:** 14 weeks (3.5 months)

---

### Milestone Map

```
Week  1 ─── [M0] Dev environment validated; baseline face detection running
Week  4 ─── [M1] Prototype pipeline: webcam → perturb → display (monolithic)
Week  6 ─── [M2] MCP server live; all 4 tools callable via JSON-RPC
Week  7 ─── [M3] Adaptive feedback loop operational; alpha scaling proven
Week  8 ─── [M4] Virtual camera output confirmed on Zoom + Teams + Meet
Week 10 ─── [M5] Attack success rate < 20%; SSIM ≥ 0.95; latency < 60 ms
Week 11 ─── [M6] Docker image published to internal registry
Week 13 ─── [M7] Jetson Nano deployment validated at < 30 ms
Week 14 ─── [M8] Documentation complete; stakeholder demo delivered
```

---

## 6. Detailed Workflow by Phase

---

### Phase 0 — Foundation & Environment Setup (Week 1)

**Objective:** Establish a fully reproducible development environment and validate all dependency chains before any code is written.

**Tasks:**

1. **Repository Setup**
   - Create the monorepo: `adversarial-face-shield/`
   - Submodule or vendor: `ReFace`, `DeepSafe`, `AdversarialMask`, `FaceOff`
   - Initialize Git with branching strategy: `main`, `develop`, `feature/*`, `hotfix/*`

2. **Python Virtual Environment**
   ```bash
   python -m venv dfs-shield
   source dfs-shield/bin/activate
   pip install torch torchvision opencv-python mediapipe dlib numpy tqdm pyvirtualcam
   pip install mcp-sdk  # or websockets for custom JSON-RPC
   ```

3. **Pretrained Model Acquisition**
   - Download `reface_atn.pth` from the official ReFace release
   - Download DeepSafe model checkpoints
   - Validate SHA-256 checksums against published hashes
   - Store models in `models/` with a `models/registry.json` manifest

4. **Hardware Verification**
   - Confirm CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
   - Benchmark baseline ATN forward pass time (GPU vs CPU)
   - Confirm webcam access via OpenCV

5. **Baseline Face Detection Test**
   - Run MediaPipe FaceMesh on a test image; verify landmark output
   - Run dlib HOG detector as fallback; compare latency

**Deliverable:** `environment_report.md` documenting all installed package versions, GPU specs, and baseline latency measurements.

**Dependencies:** NVIDIA GPU with CUDA 11.8+; Python 3.10+; Ubuntu 22.04 LTS or Windows 11 (WSL2).

---

### Phase 1 — Core Pipeline Prototype (Weeks 2–4)

**Objective:** Build a functional, monolithic proof-of-concept demonstrating the complete webcam → detect → perturb → blend → display pipeline in a single Python script.

**Tasks:**

1. **Face Detection Module** (`src/detection/face_detector.py`)
   - Wrap MediaPipe FaceMesh; expose a `detect(frame: np.ndarray) -> List[BoundingBox]` interface
   - Implement affine alignment using 5-point landmarks (eye corners, nose tip)
   - Unit-test on static images with ground truth bounding boxes

2. **ATN Inference Module** (`src/perturbation/atn_engine.py`)
   - Load `reface_atn.pth`; expose `generate(face_crop: np.ndarray) -> np.ndarray`
   - Implement preprocessing: resize to ATN input resolution (e.g., 224×224), normalize to [-1, 1]
   - Implement postprocessing: denormalize perturbation, reshape to original crop dimensions
   - Benchmark: measure forward pass time over 100 iterations; target < 15 ms on GPU

3. **Frame Blending Module** (`src/blending/frame_blender.py`)
   - Implement `blend(frame, perturbation, boxes, alpha) -> np.ndarray`
   - Use `cv2.addWeighted` for additive blending within bounding box ROI
   - Apply mild Gaussian smoothing (kernel 3×3) to the ROI boundary to avoid edge artifacts
   - Clamp output pixel values to [0, 255]

4. **Monolithic Pipeline Script** (`prototype/pipeline.py`)
   - Chain all modules in a single `while True` capture loop
   - Display output via `cv2.imshow` for real-time visual validation

5. **Visual Quality Assessment**
   - Capture 100 shielded frames; compute SSIM against unshielded originals
   - Adjust alpha until SSIM ≥ 0.95

**Deliverable:** Runnable `prototype/pipeline.py` with documented latency profiling and SSIM measurements.

---

### Phase 2 — MCP Server Integration (Weeks 5–6)

**Objective:** Decompose the monolithic prototype into discrete MCP tools, establish the server, and validate all tool calls over JSON-RPC.

**Tasks:**

1. **MCP Server Scaffold** (`mcp_server/server.py`)
   - Initialize the MCP `Server` instance with `name: "AdversarialFaceShield"`
   - Register lifecycle hooks: `initialize`, `shutdown`
   - Configure transport: `stdio` for local, HTTP for remote (configurable via `config.yaml`)

2. **Tool Implementation — `face_detector`**
   - Accept: `{ "frame_b64": "<string>" }`
   - Decode frame → run MediaPipe → return `{ "boxes": [[x1,y1,x2,y2], ...], "landmarks": [...] }`
   - Error handling: return `{ "boxes": [], "error": "no_face_detected" }` if detection fails

3. **Tool Implementation — `perturbation_generator`**
   - Accept: `{ "face_b64": "<string>" }`
   - Decode → preprocess → ATN forward pass → encode perturbation
   - Return: `{ "perturbation_b64": "<string>", "latency_ms": <float> }`

4. **Tool Implementation — `frame_blender`**
   - Accept: `{ "frame_b64": "<string>", "perturbation_b64": "<string>", "boxes": [...], "alpha": <float> }`
   - Blend → encode → return `{ "shielded_frame_b64": "<string>" }`

5. **Tool Implementation — `deepfake_feedback`** *(optional, feature-flagged)*
   - Accept: `{ "frame_b64": "<string>" }`
   - Run DeepSafe CNN-LSTM → return `{ "confidence": <float 0-1>, "label": "real"|"fake" }`

6. **Capture Client** (`mcp_client/capture_client.py`)
   - Implement the main capture loop calling tools in sequence via MCP client SDK
   - Implement adaptive alpha logic: `if confidence > 0.70: alpha = min(alpha + 0.02, 0.30)`

7. **Integration Testing**
   - Write pytest integration tests calling each MCP tool in isolation
   - Write end-to-end test simulating a 10-second video stream

**Deliverable:** Fully functional MCP server with 4 registered tools; integration test suite with ≥ 90% pass rate.

---

### Phase 3 — Feedback Loop & Adaptive Alpha (Week 7)

**Objective:** Activate the deepfake feedback tool in the live pipeline and demonstrate measurable improvement in attack resistance when confidence thresholds are exceeded.

**Tasks:**

1. **Feedback Scheduling**
   - Implement `FeedbackScheduler` class: calls `deepfake_feedback` every N frames (configurable, default N=5)
   - Maintain a rolling average of last 10 confidence scores to smooth alpha adjustments

2. **Alpha Controller**
   - Implement PID-inspired controller: increase alpha when average confidence > 0.70; decay alpha when < 0.40
   - Log all alpha adjustments with timestamps to `logs/alpha_trace.jsonl`

3. **Validation**
   - Test with a pre-recorded "high-risk" video (face clearly visible); verify alpha automatically escalates
   - Test with a blurred/obstructed face; verify alpha decays to baseline

**Deliverable:** Adaptive feedback loop demonstration video; alpha trace logs showing dynamic behavior.

---

### Phase 4 — Virtual Camera & UX Integration (Week 8)

**Objective:** Route the shielded stream to a virtual camera device and confirm compatibility with major conferencing platforms.

**Tasks:**

1. **pyvirtualcam Integration**
   - Instantiate virtual camera at target resolution (1280×720) and FPS (30)
   - Write shielded frames to virtual cam device in the capture loop
   - Handle frame-rate synchronization: `cam.sleep_until_next_frame()`

2. **Platform Compatibility Testing**
   - Test virtual cam visibility and stream quality in: Zoom, Microsoft Teams, Google Meet, OBS Studio, Webex
   - Document any platform-specific configuration requirements

3. **System Tray Application** *(minimal UI)*
   - Implement a lightweight `systray.py` using `pystray` for start/stop/status controls
   - Expose current alpha value and FPS in tray tooltip

4. **Graceful Shutdown**
   - Implement `SIGINT` and `SIGTERM` handlers to cleanly release webcam, virtual camera, and MCP connections

**Deliverable:** Working virtual camera output confirmed on ≥ 3 platforms; tray application functional.

---

### Phase 5 — Validation & Benchmarking (Weeks 9–10)

**Objective:** Formally measure all success criteria against quantitative targets.

**Tasks:**

1. **Attack Success Rate Benchmark**
   - Submit 200 shielded and 200 unshielded face images to a commercial face-recognition API (e.g., AWS Rekognition)
   - Measure verification success rate for each cohort
   - Target: unshielded ≈ 80%, shielded < 20%

2. **Visual Quality Benchmark**
   - Compute SSIM on 500 shielded/unshielded frame pairs using `skimage.metrics.structural_similarity`
   - Conduct a 20-participant MOS user study (1–5 scale); target MOS ≥ 4.0

3. **Latency Benchmark**
   - Instrument each pipeline stage with high-resolution timestamps
   - Measure P50, P95, P99 end-to-end latency over 1,000 frames
   - Target P99 < 60 ms

4. **Stress Testing**
   - Run pipeline continuously for 2 hours; monitor for memory leaks (GPU VRAM, CPU RAM)
   - Test with 3 simultaneous face detections per frame

5. **Adversarial Robustness**
   - Apply ApaNet denoising to shielded frames; measure how much attack resistance is retained

**Deliverable:** Benchmarking report (`validation/benchmark_report.md`) with all measured values vs. targets.

---

### Phase 6 — Containerization & Deployment (Week 11)

**Objective:** Package the system into reproducible, production-ready Docker containers.

**Tasks:**

1. **Dockerfile — MCP Server**
   ```dockerfile
   FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
   WORKDIR /app
   COPY mcp_server/ .
   COPY models/ ./models/
   RUN pip install -r requirements.txt
   EXPOSE 8080
   CMD ["python", "server.py", "--transport", "http"]
   ```

2. **Dockerfile — Capture Client**
   - Separate lightweight image without GPU dependencies
   - Mount host webcam device: `--device /dev/video0`

3. **Docker Compose**
   - Define `mcp-server`, `capture-client`, `feedback-service` services
   - Use named volumes for model checkpoints
   - Configure health checks for each service

4. **Kubernetes Manifests**
   - MCP server: `Deployment` with GPU resource limits + `Service` (ClusterIP)
   - Capture client: `DaemonSet` targeting workstation nodes
   - Feedback service: `Deployment` with optional HPA on CPU usage

5. **CI/CD Pipeline** (GitHub Actions)
   - Lint → Unit Tests → Integration Tests → Docker Build → Push to Registry
   - Trigger on `develop` and `main` branch pushes

**Deliverable:** Published Docker images; `docker-compose up` successfully starts the full system on a clean machine.

---

### Phase 7 — Edge Optimization (Weeks 12–13)

**Objective:** Optimize the ATN for NVIDIA Jetson Nano deployment achieving sub-30 ms inference.

**Tasks:**

1. **TensorRT Export**
   - Export ATN to ONNX: `torch.onnx.export(atn_model, dummy_input, "atn.onnx")`
   - Convert ONNX to TensorRT engine using `trtexec`
   - Validate numerical accuracy: perturbation outputs must differ by < 1% from PyTorch baseline

2. **Jetson-Specific Optimizations**
   - Use FP16 precision (`--fp16` flag in `trtexec`) for 2× throughput
   - Pin webcam to dedicated CPU core using `taskset`
   - Tune MediaPipe to use Coral/NVDLA for face detection

3. **Edge Deployment Profile**
   - Create `config/edge.yaml` with Jetson-specific parameters (lower resolution fallback: 640×480)
   - Measure and document power consumption (target: < 10W)

**Deliverable:** TensorRT engine file; documented Jetson setup guide; latency measurement < 30 ms.

---

### Phase 8 — Documentation & Handover (Week 14)

**Objective:** Produce complete technical and operational documentation for the system.

**Tasks:**

1. **Technical Documentation**
   - System architecture document with updated diagrams
   - API reference for all MCP tools (input schemas, output schemas, error codes)
   - Configuration reference (`config.yaml` all parameters)

2. **Operational Runbooks**
   - Installation guide (Docker, bare metal, Jetson)
   - Troubleshooting guide (common issues, diagnostic commands)
   - Upgrade / rollback procedures

3. **Stakeholder Presentation**
   - Live demonstration of AFS on video call
   - Results summary: attack success rate, SSIM, latency benchmarks

4. **Knowledge Transfer**
   - 2-hour technical walkthrough session with engineering team
   - Record session for async reference

**Deliverable:** All documentation published to internal wiki; stakeholder demo completed.

---

## 7. Roles & Responsibilities

| Role | Responsibilities | Phase Involvement |
|---|---|---|
| **AI Research Lead** | ATN selection, perturbation tuning, SSIM/attack-success validation, feedback loop design | 0, 1, 3, 5 |
| **ML Engineer** | ATN inference optimization, TensorRT export, GPU profiling | 0, 1, 2, 7 |
| **Backend Engineer** | MCP server implementation, JSON-RPC protocol, tool registration, error handling | 2, 3, 4 |
| **Computer Vision Engineer** | Face detection pipeline, landmark alignment, OpenCV frame handling, virtual camera integration | 0, 1, 4 |
| **DevOps / MLOps Engineer** | Docker/Kubernetes, CI/CD, container registry, Jetson deployment | 6, 7 |
| **QA Engineer** | Test suite design, benchmark execution, stress testing, MOS user study coordination | 5 |
| **Project Manager** | Sprint planning, milestone tracking, stakeholder communication, risk management | All |
| **Security Reviewer** | MCP security model review, authentication controls, API key management for AWS Rekognition benchmarks | 2, 5 |

### Dependency Map

```
Phase 0 → Phase 1 (environment must be stable before code is written)
Phase 1 → Phase 2 (modules must be individually validated before MCP wrapping)
Phase 2 → Phase 3 (MCP tools must be stable before feedback loop is added)
Phase 2 → Phase 4 (MCP client must be functional before virtual cam integration)
Phase 3 + Phase 4 → Phase 5 (full pipeline must be complete before benchmarking)
Phase 5 → Phase 6 (validation must pass before production containerization)
Phase 6 → Phase 7 (Docker baseline must exist before edge optimization)
Phase 5 + Phase 6 + Phase 7 → Phase 8 (all results needed for documentation)
```

---

## 8. Integration Points & Key Functionality

### 8.1 MCP Tool Interface Contracts

All tool contracts are defined in `mcp_server/schemas/` as JSON Schema documents. Any changes to tool input/output schemas must go through the schema versioning process (SemVer minor bump) to avoid breaking clients.

**`face_detector` Contract:**
```json
{
  "input":  { "frame_b64": "string" },
  "output": {
    "boxes":     [["number", "number", "number", "number"]],
    "landmarks": [["number", "number"]],
    "error":     "string | null"
  }
}
```

**`perturbation_generator` Contract:**
```json
{
  "input":  { "face_b64": "string" },
  "output": { "perturbation_b64": "string", "latency_ms": "number" }
}
```

**`frame_blender` Contract:**
```json
{
  "input": {
    "frame_b64": "string",
    "perturbation_b64": "string",
    "boxes": [["number", "number", "number", "number"]],
    "alpha": "number"
  },
  "output": { "shielded_frame_b64": "string" }
}
```

### 8.2 Virtual Camera Compatibility Layer

The virtual camera output must comply with the V4L2 spec on Linux. On Windows, OBS-VirtualCam driver must be installed before the capture client starts. A compatibility check function (`src/utils/virt_cam_check.py`) runs at startup and raises a descriptive error if prerequisites are unmet.

### 8.3 Alpha Control Protocol

The alpha value is a shared mutable state managed by the Capture Client's `AlphaController`. It is never modified by MCP tools directly. This separation of concerns ensures MCP tools remain stateless and reusable.

### 8.4 Model Registry

All model checkpoints are registered in `models/registry.json`:
```json
{
  "atn": { "path": "models/reface_atn.pth", "sha256": "...", "input_shape": [1,3,224,224] },
  "deepfake_detector": { "path": "models/deepsafe.pth", "sha256": "..." }
}
```
The server validates checksums at startup and refuses to start if any model file is corrupted or missing.

### 8.5 Logging & Observability

- **Structured logging:** All components emit JSON logs to `logs/` with fields: `timestamp`, `component`, `level`, `message`, `latency_ms`
- **Metrics:** Frame rate, per-stage latency, alpha value, and feedback confidence are emitted as Prometheus metrics on port 9090
- **Alerting:** Alert if P95 latency > 100 ms for 10 consecutive frames (logged as WARNING; triggers alpha decay to reduce compute load)

---

## 9. Risk Register & Mitigation Strategies

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | **ATN perturbations become perceptible at required alpha** | Medium | High | Tune alpha per-scene (dark vs. bright environments); implement skin-tone-aware blending; evaluate alternative ATN checkpoints |
| R2 | **MCP JSON-RPC overhead introduces unacceptable latency** | Medium | High | Profile and switch to shared-memory IPC if stdio latency > 20 ms; pre-allocate buffers; reduce Base64 overhead with binary framing |
| R3 | **GPU VRAM exhaustion during concurrent face processing** | Medium | Medium | Cap concurrent ATN inferences at N=3; implement face-ROI queue with backpressure; use FP16 to halve VRAM usage |
| R4 | **Virtual camera driver incompatibility with target platform** | Medium | Medium | Maintain platform-specific driver installation scripts; test on Windows 11, Ubuntu 22.04, and macOS 14 |
| R5 | **DeepSafe feedback model mis-classifies legitimate video** | Low | Medium | Make feedback loop optional (config flag); implement confidence smoothing to prevent alpha oscillation |
| R6 | **ReFace ATN checkpoint becomes unavailable or deprecated** | Low | High | Vendor the checkpoint in the internal model registry; document training procedure for re-training if needed |
| R7 | **Adversarial perturbations stripped by video codec compression** | High | High | Test with H.264/H.265 compressed streams; if perturbations are lost, shift to frequency-domain perturbations more robust to DCT-based compression |
| R8 | **Edge Jetson deployment fails TensorRT conversion** | Medium | Medium | Maintain PyTorch CPU fallback; test TensorRT conversion in CI on a Jetson-spec Docker image |
| R9 | **Privacy / legal concerns about modifying user video streams** | Low | High | Add explicit user consent prompt at startup; document that modifications are opt-in; consult legal counsel before public deployment |
| R10 | **MCP server single point of failure** | Low | High | Implement health-check endpoint; auto-restart via Docker restart policy; document manual recovery procedure |

---

## 10. Quality Assurance & Validation Framework

### 10.1 Test Levels

| Level | Scope | Tool | Trigger |
|---|---|---|---|
| Unit | Individual functions (detect, perturb, blend) | pytest | Every commit |
| Integration | MCP tool calls end-to-end | pytest + mcp-client mock | Every PR |
| System | Full pipeline on live video | Manual + automated script | Every milestone |
| Performance | Latency, SSIM, attack rate | Custom benchmark scripts | Phases 5 & 7 |
| Security | MCP auth, payload injection | Manual security review | Phase 2 & pre-launch |

### 10.2 Definition of Done (Per Phase)

A phase is complete when:
1. All tasks are merged to `develop` with peer-reviewed PRs
2. Unit and integration test suites pass at ≥ 95%
3. Phase deliverable is accepted by Project Manager and AI Research Lead
4. Benchmark results (where applicable) meet defined targets

### 10.3 Continuous Integration Gates

Pull requests to `develop` must pass:
- `flake8` lint (zero errors)
- `mypy` static type check (zero errors on typed modules)
- Full pytest suite (≥ 95% pass)
- Docker build succeeds (Phases 6+)

---

## 11. Deployment Strategy

### 11.1 Local Developer Deployment

```bash
git clone https://github.com/your-org/adversarial-face-shield.git
cd adversarial-face-shield
python -m venv dfs-shield && source dfs-shield/bin/activate
pip install -r requirements.txt
# Download models
python scripts/download_models.py
# Start MCP server (stdio mode)
python mcp_server/server.py &
# Start capture client
python mcp_client/capture_client.py --alpha 0.12 --feedback-interval 5
```

### 11.2 Docker Deployment

```bash
docker-compose up --build
# MCP server starts on port 8080 (HTTP transport)
# Capture client connects via HTTP
# Virtual cam device: /dev/video10 (Linux)
```

### 11.3 Kubernetes Deployment

```bash
kubectl apply -f k8s/mcp-server-deployment.yaml
kubectl apply -f k8s/capture-daemonset.yaml
kubectl apply -f k8s/feedback-deployment.yaml
# Monitor
kubectl logs -f deployment/mcp-server
```

### 11.4 Jetson Nano Deployment

```bash
# On Jetson (JetPack 5.1+)
docker pull your-registry/afs-jetson:latest
docker run --runtime nvidia --device /dev/video0 --device /dev/video10 \
  -e AFS_CONFIG=config/edge.yaml \
  your-registry/afs-jetson:latest
```

### 11.5 Rollback Procedure

If a deployment fails validation post-release:
1. Revert to previous Docker image tag: `docker-compose down && docker-compose up --image previous-tag`
2. Restore model registry to previous `registry.json` snapshot
3. Notify stakeholders via incident channel within 15 minutes
4. Root-cause analysis document due within 48 hours

---

## 12. Appendices

### Appendix A — Repository Structure

```
adversarial-face-shield/
├── mcp_server/
│   ├── server.py              # MCP server entrypoint
│   ├── tools/
│   │   ├── face_detector.py
│   │   ├── perturbation_generator.py
│   │   ├── frame_blender.py
│   │   └── deepfake_feedback.py
│   └── schemas/               # JSON Schema contracts
├── mcp_client/
│   ├── capture_client.py      # Main capture + orchestration loop
│   ├── alpha_controller.py    # PID-inspired alpha control
│   └── virt_cam_driver.py     # pyvirtualcam wrapper
├── src/
│   ├── detection/
│   ├── perturbation/
│   └── blending/
├── models/
│   └── registry.json
├── config/
│   ├── default.yaml
│   └── edge.yaml
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmark/
├── k8s/
├── scripts/
├── logs/
├── docker-compose.yml
├── Dockerfile.server
├── Dockerfile.client
└── README.md
```

### Appendix B — Key Configuration Parameters (`config/default.yaml`)

```yaml
pipeline:
  fps_target: 30
  resolution: [1280, 720]
  alpha_initial: 0.12
  alpha_max: 0.30
  alpha_decay_rate: 0.02
  feedback_interval_frames: 5
  feedback_confidence_threshold: 0.70

detection:
  backend: mediapipe          # or: dlib
  min_detection_confidence: 0.7

perturbation:
  atn_input_size: 224
  device: cuda                # or: cpu

transport:
  mode: stdio                 # or: http
  http_port: 8080

logging:
  level: INFO
  output: logs/afs.jsonl
  metrics_port: 9090
```

### Appendix C — Reference Codebases & Papers

| Resource | URL / Reference |
|---|---|
| ReFace ATN | `github.com/ReFace-attack/ReFace` |
| DeepSafe | `github.com/siddharthksah/DeepSafe` |
| FaceOff | `github.com/392781/FaceOff` |
| AdversarialMask | `github.com/AlonZolfi/AdversarialMask` |
| Awesome Deepfakes Detection | `github.com/Daisy-Zhang/Awesome-Deepfakes-Detection` |
| MCP Specification | `modelcontextprotocol.io/specification/2025-03-26` |
| ReFace Paper | arxiv.org/pdf/2206.04783.pdf |
| ApaNet | PMC9395815 |

### Appendix D — Glossary

| Term | Definition |
|---|---|
| **ATN** | Adversarial Transformation Network — a feed-forward neural network that maps an input face to an adversarial perturbation in a single pass |
| **Alpha (α)** | Blending coefficient controlling the intensity of the adversarial perturbation applied to the frame |
| **MCP** | Model Context Protocol — a standard for exposing AI model capabilities as tools via JSON-RPC 2.0 |
| **ROI** | Region of Interest — the bounding box subimage containing a detected face |
| **SSIM** | Structural Similarity Index — a perceptual image quality metric (1.0 = identical) |
| **MOS** | Mean Opinion Score — a subjective video quality rating on a 1–5 scale |
| **TensorRT** | NVIDIA's inference optimization library for deploying deep learning models on NVIDIA hardware |
| **Virtual Camera** | A software-emulated webcam device that presents a modified video stream to applications |
| **pyvirtualcam** | Python library for writing frames to a virtual camera device |

---

*This document is a living artifact. It should be updated at each phase completion to reflect actual versus planned progress, revised risk assessments, and any architectural decisions made during implementation.*

---

**Document End — Adversarial Face Shield Project Plan v1.0**
