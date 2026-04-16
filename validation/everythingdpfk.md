# EverythingDPFK

## Implementation Record: NSFW-Trigger Extension (2026-04-17)

This document tracks exactly how the NSFW-trigger capability was implemented in the existing AFS pipeline, what was executed to verify it, observed runtime behavior, and concrete fixes/fallbacks that were required to reach a stable end-to-end run.

The focus here is implementation and runtime execution only.

## 1) End-to-End Implementation Topology

The existing 4-tool chain was extended without breaking default shield behavior:

- Existing chain retained:
  - face_detector
  - perturbation_generator
  - frame_blender
  - deepfake_feedback
- New NSFW components added:
  - nsfw_feedback JSON-RPC method (new tool)
  - dual perturbation profiles in perturbation_generator:
    - shield_only
    - nsfw_trigger_only
    - shield_and_nsfw

The implementation keeps shield_only as the backward-compatible default path, and only activates dual-head behavior when a non-default protection profile is requested.

## 2) Backend Tooling and Contracts

### 2.1 perturbation_generator contract extension

Implemented in:

- mcp_server/schemas/perturbation_generator.schema.json
- mcp_server/tools/perturbation_generator.py

Behavior:

- Added optional protection_profile enum to RPC params.
- If protection_profile is omitted, server behaves as before (shield_only).
- For shield_only, existing ATNEngine.generate path remains unchanged.
- For nsfw_trigger_only or shield_and_nsfw, handler routes into a singleton DualHeadATNEngine.

Response shape remains compatible and now includes selected profile.

### 2.2 nsfw_feedback tool

Implemented in:

- mcp_server/tools/nsfw_feedback.py
- mcp_server/schemas/nsfw_feedback.schema.json
- mcp_server/server.py

Behavior:

- New JSON-RPC method nsfw_feedback accepts frame_b64.
- Returns JSON-safe payload:
  - nsfw_score (float in [0,1])
  - label (safe or nsfw_flagged)
  - proxies_used
- Graceful error payload for malformed inputs or runtime exceptions.

### 2.3 server registration and dispatch

Implemented in:

- mcp_server/server.py

Behavior:

- Added nsfw_feedback to dispatch table.
- Kept auth/rate-limiting/schema-validation flow unchanged.
- Passed full runtime config into perturbation_generator and nsfw_feedback handlers so both can resolve nsfw_trigger settings consistently.

## 3) NSFW Feedback Engine Implementation

Implemented in:

- src/feedback/nsfw_feedback_engine.py

Key implementation decisions:

- Uses an ensemble wrapper NSFWProxyEnsemble.
- Current proxy registry includes Falconsai model via HuggingFace.
- Loads processor/model once and caches in memory.
- Resolves NSFW class index from model label map by keyword matching.
- Uses geometric mean across proxies (currently one proxy) for stable score aggregation.

Runtime path:

1. Decode frame_b64 into PIL RGB image.
2. Convert to float tensor [0,1], shape (B,C,H,W).
3. Resize to proxy target input size.
4. Normalize with processor image mean/std.
5. Compute softmax and extract NSFW probability.
6. Return rounded score and thresholded label.

Label threshold currently uses 0.5:

- score >= 0.5 -> nsfw_flagged
- score < 0.5 -> safe

## 4) NSFW Trigger Perturbation Implementation

### 4.1 NSFWTriggerATN model

Implemented in:

- src/perturbation/nsfw_trigger_atn.py

Model shape:

- Encoder-decoder with residual bottleneck.
- Input/output tensors are in [0,1].
- Bounded perturbation via l_inf_bound scaling and clamping.

Important runtime safety added:

- If decoder output spatial size does not match input size, output is resized back to exact input HxW before delta clamp.

This fix removed the combined-mode tensor mismatch failure seen on odd ROI sizes during smoke execution.

### 4.2 Dual-head engine composition

Implemented in:

- src/perturbation/atn_engine.py (DualHeadATNEngine)
- src/perturbation/perturbation_combiner.py

Composition details:

- Shield head: existing reface ATN path.
- NSFW head: NSFWTriggerATN.
- Combined profile: weighted sum of shield delta and nsfw delta, then joint L-infinity cap and output clamp.
- SSIM-proxy guard reduces NSFW contribution automatically if quality proxy falls below configured floor.

This keeps imperceptibility constraints active while allowing NSFW signal injection.

## 5) Frontend Integration

Implemented in:

- frontend/src/services/pipelineService.ts
- frontend/src/hooks/usePipelineRunner.ts
- frontend/src/services/rpcClient.ts
- frontend/src/types/api.ts

Integration details:

- Added protectionProfile plumbing from UI runner to perturbation_generator RPC.
- Added runNsfwFeedback helper and nsfw_score propagation in pipeline results.
- Preserved existing behavior when profile is shield_only and nsfw feedback is disabled.
- Kept stream safety with in-flight guard and skipped-frame counting.

Payload behavior:

- perturbation_generator receives padded ROI crop JPEG.
- frame_blender continues receiving full frame and original boxes.

## 6) Config, Models, and Validation Hooks

Implemented in:

- config/default.yaml
- config/smoke.yaml
- config/edge.yaml
- models/registry.json
- scripts/preflight_check.py
- scripts/update_registry_hashes.py

Configuration path:

- nsfw_trigger block includes enabled, alpha, device, proxies, and profile defaults.
- models.nsfw_trigger_atn.path is wired in smoke profile.
- strict_startup false and allow_identity_fallback true in smoke profile preserve local bring-up robustness.

Proxy weight provisioning:

- scripts/download_nsfw_proxies.py downloads and caches Falconsai proxy artifacts into models/nsfw_proxy.

## 7) Execution and Verification Performed

The following commands were executed in the local workspace to validate implementation behavior.

### 7.1 NSFW trigger unit tests

Command:

- .\.venv\Scripts\python.exe -m pytest tests/unit/test_nsfw_trigger_atn.py -q

Result:

- 4 passed in 2.41s

Coverage verified:

- output shape consistency
- L-infinity bound enforcement
- output clamp correctness
- architecture signature presence

### 7.2 NSFW feedback integration test

Command:

- .\.venv\Scripts\python.exe -m pytest tests/integration/test_nsfw_feedback.py -q

Result:

- 1 passed in 15.29s

Coverage verified:

- live server startup from temporary smoke config
- nsfw_feedback RPC response contract
- nsfw_score numeric range [0,1]

### 7.3 End-to-end smoke chain with NSFW path

Command:

- .\.venv\Scripts\python.exe scripts/smoke_test_http_pipeline.py --python .venv/Scripts/python.exe --config config/smoke.yaml --api-key afs-local-dev-key

Result payload (current run):

- status: ok
- tools_called:
  - face_detector
  - perturbation_generator
  - frame_blender
  - deepfake_feedback
  - nsfw_feedback
  - perturbation_generator(shield_and_nsfw)
- deepfake_feedback:
  - confidence: 0.5360739988221854
  - label: fake
- nsfw_feedback:
  - nsfw_score: 0.012855
  - label: safe
  - proxies_used: [falconsai]
- nsfw_profile: shield_and_nsfw

Interpretation:

- NSFW-trigger pipeline is wired and callable end-to-end.
- Processed frame can be evaluated by nsfw_feedback immediately after blending.
- In this run, processed frame remained in safe band for the configured proxy/threshold.

## 8) Performance and Runtime Behavior

Observations from this implementation cycle:

- Local smoke chain completed within the scripted per-call timeout budget (30s).
- NSFW feedback path adds an additional model inference stage after blending; it is functionally stable under CPU smoke profile.
- Deepfake feedback confidence remained near baseline (~0.536) during combined profile smoke call, indicating no immediate regressions in RPC chain behavior.

Current local benchmark artifact remains available in validation/benchmark_results.json for broader latency/quality tracking.

## 9) Fallbacks, Fixes, and Corrections Applied

### 9.1 Critical fix: odd-size ROI tensor mismatch

Failure observed:

- Runtime error in combined mode due to shape mismatch (example: width 639 vs 640).

Correction:

- Added spatial interpolation in NSFWTriggerATN.forward to force raw decoder output to match input spatial size before combining deltas.

Result:

- Combined profile (shield_and_nsfw) smoke call now succeeds.

### 9.2 Proxy loader/API correction

Failure observed:

- Proxy download/usage path initially mismatched HuggingFace processor APIs.

Correction:

- Standardized on AutoImageProcessor and corresponding normalization path used by runtime scorer.

Result:

- Proxy load path is consistent across downloader and runtime scoring.

### 9.3 Startup and smoke reliability corrections

Corrections applied:

- smoke script resolves endpoint and auth from CLI/config/env in deterministic order.
- smoke logs mask API key values.
- startup health probing and local server lifecycle handling were hardened.

Result:

- Repeatable smoke behavior with reduced config drift and clearer diagnostics.

### 9.4 Backward-compatibility fallback behavior preserved

Preserved behaviors:

- shield_only remains default profile with existing path unchanged.
- no requirement to enable NSFW profile for legacy clients.
- strict_startup and identity-fallback behavior in ATN load path still protects bring-up in non-strict profiles.

## 10) Current Implementation Status

Implemented and verified in code and execution:

- NSFWTriggerATN head and combiner path
- perturbation_generator profile routing
- nsfw_feedback RPC method and schema
- frontend profile and nsfw score plumbing
- smoke script NSFW chain execution
- unit and integration verification for NSFW components

Remaining operational dependency for full training validation:

- Trained checkpoint quality still depends on available training data and checkpoint generation workflow.
