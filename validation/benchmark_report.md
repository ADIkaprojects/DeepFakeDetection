# Phase 5 Benchmark Report

Date: 2026-04-07

Environment:
- OS: Windows
- Python: 3.13 (workspace venv)
- Config used: config/smoke.yaml
- Runtime mode: local module benchmark plus authenticated HTTP smoke

Method Summary:
- End-to-end HTTP smoke executed through all tools: face_detector, perturbation_generator, frame_blender, deepfake_feedback.
- Quantitative benchmark executed with 200 synthetic frames using local module pipeline.
- SSIM computed with scikit-image.

Targets:
- Attack success rate (shielded): < 20%
- SSIM: >= 0.95
- End-to-end latency: < 60 ms

Measured Results:
- P50 latency: 3.81 ms
- P95 latency: 4.40 ms
- P99 latency: 4.77 ms
- Mean latency: 3.82 ms
- SSIM mean: 0.98996
- SSIM min/max: 0.98996 / 0.98996

Smoke Test Result:
- Status: PASS
- Output artifact: validation/smoke_output.png
- RPC artifact: validation/smoke_result.json

Deferred Metrics (not executable in this local environment):
- MOS user study: pending participant study setup.
- Commercial face-recognition attack benchmark: pending credentialed API integration and dataset run.
- Attack success rate (unshielded/shielded): pending external benchmark execution.

Gap Notes:
- MediaPipe FaceMesh API is unavailable in current installed package variant (`mediapipe.solutions` missing), so detector is in graceful degraded mode.
- Real ReFace and DeepSafe checkpoints are not yet present in models directory for strict production-mode startup.

Next Required Actions for Full Phase 5 Completion:
1. Place real ReFace and DeepSafe checkpoints and update SHA256 values in models/registry.json.
2. Re-run benchmark in strict mode (`models.strict_startup: true`) with real checkpoints.
3. Execute API-based attack success benchmark on 200 shielded and 200 unshielded frames.
4. Run MOS study (n=20+) and append confidence intervals.
