# AFS × NSFW-Trigger: Full Implementation Plan
## "Pixel Immunity — Deepfake Immune System, Phase 2"

> **Goal**: Extend the existing Adversarial Face Shield (AFS) pipeline with a new, stackable
> `nsfw_trigger` perturbation mode. When a user's photo is uploaded to ChatGPT Image Editor,
> Nano Banana, Stable Diffusion, or any similar AI platform, the injected invisible perturbation
> causes the platform's own NSFW/safety classifier to flag the image as explicit — making it
> reject, quarantine, or refuse to train on the photo — while a human observer sees nothing
> abnormal whatsoever.

---

## Architecture Overview (What We Are Building)

```
Existing AFS pipeline
┌───────────────────────────────────────────────────────┐
│  face_detector → perturbation_generator → frame_blender → deepfake_feedback │
└───────────────────────────────────────────────────────┘

After this plan
┌─────────────────────────────────────────────────────────────────────────────────┐
│  face_detector                                                                   │
│       ↓                                                                          │
│  perturbation_generator (dual-head)                                              │
│       ├── Head 1: ReFaceATN        → deepfake/recognition shield (existing)     │
│       └── Head 2: NSFWTriggerATN   → NSFW-flag injection (NEW)                  │
│       ↓                                                                          │
│  perturbation_combiner (NEW: weighted sum + clamping)                            │
│       ↓                                                                          │
│  frame_blender (extended: per-head alpha + joint SSIM guard)                    │
│       ↓                                                                          │
│  dual_feedback (extended)                                                        │
│       ├── deepfake_feedback → fake confidence (existing)                         │
│       └── nsfw_feedback (NEW) → NSFW score from proxy classifiers               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

New files we will create:
```
src/perturbation/nsfw_trigger_atn.py       ← NSFWTriggerATN model
src/perturbation/perturbation_combiner.py  ← dual-head combiner
src/feedback/nsfw_feedback_engine.py       ← NSFW proxy feedback
src/training/nsfw_trigger_trainer.py       ← training loop
scripts/train_nsfw_trigger.py              ← training CLI
scripts/validate_nsfw_trigger.py           ← standalone validator
config/nsfw_trigger.yaml                   ← dedicated profile
models/nsfw_proxy/                         ← proxy classifier weights
tests/unit/test_nsfw_trigger_atn.py
tests/unit/test_perturbation_combiner.py
tests/integration/test_nsfw_feedback.py
```

Files we will extend:
```
src/perturbation/atn_engine.py             ← multi-head dispatch
mcp_server/tools/perturbation_generator.py ← expose nsfw_trigger head
mcp_server/tools/deepfake_feedback.py      ← add nsfw feedback branch
mcp_server/server.py                       ← register new tools
src/utils/model_registry.py                ← register nsfw_trigger_atn
models/registry.json                       ← add nsfw_proxy entries
config/default.yaml / smoke.yaml / edge.yaml ← add nsfw_trigger block
frontend/src/services/pipelineService.ts   ← pass protection_profile
frontend/src/hooks/usePipelineRunner.ts    ← surface NSFW score in UI
scripts/preflight_check.py                 ← check nsfw proxy models
scripts/smoke_test_http_pipeline.py        ← extend smoke chain
```

---

## STEP 0 — Read Everything: Full Codebase Audit (Do This Before Writing A Single Line)

This is not optional. Every subsequent step depends on understanding the exact interfaces,
shapes, and contracts that already exist. Before touching any file, perform this full read.

### 0.1 — Primary source files to read completely, in order

#### Group A: Detection layer
```
src/detection/face_detector.py
```
Read and note:
- The exact return shape of the `detect()` method (boxes format: [x1,y1,x2,y2] or xywh?)
- The fallback error string value: `mediapipe_unavailable_full_frame_fallback`
- The `check_mediapipe()` signature (used in preflight)
- Which exceptions are raised vs. which return graceful error dicts

#### Group B: Perturbation layer
```
src/perturbation/atn_engine.py
```
Read and note:
- The `ReFaceATN` class: `__init__` signature, `forward(face_tensor)` input/output tensor shapes
- `arch_signature()` and `compare_arch_signatures()` implementation
- `load_checkpoint()`: how `filtered_load` works, what `is_identity_mode()` means
- The exact `nn.Module` layers (encoder depth, channel sizes, activation types) — you need this
  to replicate or subclass NSFWTriggerATN with a compatible interface

#### Group C: Blending layer
```
src/blending/   (all files)
```
Read and note:
- How `blend_roi(frame, perturbation, box, alpha)` merges the perturbation back
- Gaussian smoothing parameters at ROI edges
- The pixel clamping range (uint8 [0,255] or float [-1,1]?)

#### Group D: Feedback layer
```
src/feedback/deepsafe_engine.py
```
Read and note:
- `BaseFeedbackEngine` abstract interface (methods: `predict(frame_b64)` → dict with `confidence`, `label`)
- `LightweightDeepSafeEngine` implementation details
- `UFDDeepSafeAdapter` — how it wraps the model call
- `build_feedback_engine(config)` factory: what config keys it reads

#### Group E: MCP server tools — read every tool handler
```
mcp_server/tools/face_detector.py
mcp_server/tools/perturbation_generator.py
mcp_server/tools/frame_blender.py
mcp_server/tools/deepfake_feedback.py
mcp_server/server.py
```
Read and note for EACH tool:
- The JSON-RPC method name (string)
- The exact JSON params schema it expects
- The exact JSON result schema it returns
- How it loads/caches its model (singleton? per-call?)
- Where `x-api-key` auth is enforced

#### Group F: Config system
```
config/default.yaml
config/smoke.yaml
config/edge.yaml
```
Read and note:
- Top-level config keys (transport, models, feedback, etc.)
- Exactly how `build_feedback_engine` reads from the config dict
- The `use_ufd_backend` key location and its default

#### Group G: Model registry
```
src/utils/model_registry.py
models/registry.json
scripts/update_registry_hashes.py
scripts/preflight_check.py
scripts/verify_checkpoint.py
```
Read and note:
- Canonical required fields per registry entry: `name`, `path`, `sha256`, `architecture`, `exported_at`
- How `validate_registry()` works
- How `preflight_check.py` reports schema issues vs. missing file issues

#### Group H: Frontend service layer
```
frontend/src/services/pipelineService.ts
frontend/src/hooks/usePipelineRunner.ts
```
Read and note:
- The `cropFaceRegion()` helper: input/output types, canvas path
- How the RPC endpoint and API key are read from env
- The exact JSON body shape sent to `perturbation_generator`
- Where the `in_flight` guard lives and how `skipped_frame_counter` is used

#### Group I: Tests
```
tests/conftest.py
pytest.ini
tests/unit/   (all existing)
tests/integration/  (all existing)
tests/e2e/pipeline_happy_path.spec.ts
tests/e2e/pipeline_error_handling.spec.ts
```
Read and note:
- Fixture names available in conftest (so new tests can reuse them)
- How the integration test fires an HTTP RPC call end-to-end
- How Playwright intercepts routes (so you can add NSFW tool interception)

### 0.2 — External library API audit

Before writing any new model code, confirm available API surface:

```python
# Run this once to confirm your environment
import torch; print(torch.__version__)
import torchvision; print(torchvision.__version__)

# Confirm timm availability (for NSFW proxy backbone)
try:
    import timm; print("timm:", timm.__version__)
except ImportError:
    print("timm not installed — add to requirements.txt")

# Confirm clip availability
try:
    import clip; print("CLIP available")
except ImportError:
    print("clip not installed")

# Confirm transformers (for NudeNet / LAION NSFW proxy)
try:
    import transformers; print("transformers:", transformers.__version__)
except ImportError:
    print("transformers not installed")
```

### 0.3 — Audit Checklist (complete before proceeding)

After reading all files above, fill this checklist mentally before writing any code:

- [ ] I know the exact tensor shape that `ReFaceATN.forward()` returns
- [ ] I know what `blend_roi()` expects as pixel range (float vs uint8)
- [ ] I know every key in the JSON result of `perturbation_generator` RPC
- [ ] I know how `build_feedback_engine()` instantiates from config
- [ ] I know the canonical registry entry fields
- [ ] I know which conftest fixtures are available for reuse

---

## STEP 1 — Install and Pin New Dependencies

### 1.1 — Add to `requirements.txt`

```
# NSFW proxy classifiers
timm==0.9.16
transformers==4.40.1
Pillow==10.3.0           # already present most likely — pin if not

# Training support (if not already present)
tqdm==4.66.4
torchmetrics==1.4.0
scikit-image==0.24.0     # for SSIM computation during training
```

### 1.2 — Install

```powershell
.\.venv\Scripts\python.exe -m pip install timm==0.9.16 transformers==4.40.1 torchmetrics==1.4.0 scikit-image==0.24.0 tqdm==4.66.4
```

### 1.3 — Download NSFW Proxy Model Weights

We use **two proxy NSFW classifiers** to get transferability across different platforms.

**Proxy A — Falconsai NSFW image detection (HuggingFace)**
This is a ViT-based classifier fine-tuned on NSFW/SFW datasets.

```python
# scripts/download_nsfw_proxies.py
"""
Download and cache NSFW proxy classifier weights into models/nsfw_proxy/
Run once: python scripts/download_nsfw_proxies.py
"""
from pathlib import Path
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch

PROXY_DIR = Path("models/nsfw_proxy")
PROXY_DIR.mkdir(parents=True, exist_ok=True)

print("[1/2] Downloading Falconsai NSFW ViT proxy...")
extractor = AutoFeatureExtractor.from_pretrained(
    "Falconsai/nsfw_image_detection",
    cache_dir=str(PROXY_DIR / "falconsai_cache")
)
model_a = AutoModelForImageClassification.from_pretrained(
    "Falconsai/nsfw_image_detection",
    cache_dir=str(PROXY_DIR / "falconsai_cache")
)
model_a.eval()
torch.save(model_a.state_dict(), PROXY_DIR / "falconsai_nsfw_vit.pth")
print("   Saved to models/nsfw_proxy/falconsai_nsfw_vit.pth")

print("[2/2] Proxy download complete.")
print("   Note: LAION CLIP NSFW head is loaded on-the-fly via transformers during training.")
```

```powershell
python scripts/download_nsfw_proxies.py
```

---

## STEP 2 — Build the NSFW Proxy Ensemble (`src/feedback/nsfw_feedback_engine.py`)

This module wraps both proxy classifiers into a single interface. It is used:
1. During training of the NSFWTriggerATN (as the loss signal)
2. At runtime as a new `nsfw_feedback` MCP tool

```python
# src/feedback/nsfw_feedback_engine.py
"""
NSFW Proxy Feedback Engine.
Wraps multiple open-source NSFW classifiers into a single ensemble interface.
Used both as training signal for NSFWTriggerATN and as runtime feedback tool.
"""

from __future__ import annotations
import base64
import io
import logging
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

logger = logging.getLogger(__name__)

PROXY_REGISTRY = {
    "falconsai": {
        "hf_id": "Falconsai/nsfw_image_detection",
        "cache_dir": "models/nsfw_proxy/falconsai_cache",
        "nsfw_label_index": None,   # resolved at load time
    },
}


class NSFWProxyEnsemble:
    """
    Ensemble of open-source NSFW proxy classifiers.
    Returns a single aggregated NSFW score in [0, 1].
    Higher score = more likely the image is classified as NSFW by the proxy.
    """

    def __init__(self, device: str = "cpu", proxies: List[str] = None):
        self.device = torch.device(device)
        self.proxies_to_load = proxies or list(PROXY_REGISTRY.keys())
        self._models: Dict[str, dict] = {}
        self._load_proxies()

    def _load_proxies(self):
        for name in self.proxies_to_load:
            cfg = PROXY_REGISTRY.get(name)
            if cfg is None:
                logger.warning(f"Unknown proxy: {name}, skipping.")
                continue
            try:
                logger.info(f"Loading NSFW proxy: {name} ...")
                extractor = AutoFeatureExtractor.from_pretrained(
                    cfg["hf_id"], cache_dir=cfg["cache_dir"]
                )
                model = AutoModelForImageClassification.from_pretrained(
                    cfg["hf_id"], cache_dir=cfg["cache_dir"]
                )
                model.eval()
                model.to(self.device)

                # Resolve NSFW label index
                nsfw_idx = None
                for idx, lbl in model.config.id2label.items():
                    if lbl.lower() in ("nsfw", "explicit", "porn", "unsafe"):
                        nsfw_idx = int(idx)
                        break
                # Fallback: assume index 1 = nsfw for binary classifiers
                if nsfw_idx is None:
                    nsfw_idx = 1
                    logger.warning(
                        f"[{name}] Could not resolve NSFW label index, "
                        f"defaulting to index 1. Labels: {model.config.id2label}"
                    )

                self._models[name] = {
                    "extractor": extractor,
                    "model": model,
                    "nsfw_idx": nsfw_idx,
                }
                logger.info(f"[{name}] Loaded. NSFW label index = {nsfw_idx}")

            except Exception as e:
                logger.error(f"Failed to load NSFW proxy [{name}]: {e}")

    def score_tensor(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """
        Given a float32 face tensor of shape (B, 3, H, W) in range [0, 1],
        returns NSFW probability tensor of shape (B,).
        Used during ATN training for gradient flow.
        """
        if not self._models:
            raise RuntimeError("No NSFW proxy models loaded.")

        all_scores = []
        for name, m_dict in self._models.items():
            model = m_dict["model"]
            nsfw_idx = m_dict["nsfw_idx"]
            # Re-scale to [0, 255] uint8 range for ViT extractor expectations
            # but keep as tensor for gradient tracking
            # We pass directly as pixel_values after manual normalization
            # (ViT expects normalized pixel_values)
            extractor = m_dict["extractor"]
            img_mean = torch.tensor(extractor.image_mean, device=self.device).view(1, 3, 1, 1)
            img_std = torch.tensor(extractor.image_std, device=self.device).view(1, 3, 1, 1)

            # Resize to model expected size
            h = w = extractor.size.get("height", 224)
            x_resized = F.interpolate(face_tensor, size=(h, w), mode="bilinear", align_corners=False)
            x_norm = (x_resized - img_mean) / img_std

            logits = model(pixel_values=x_norm).logits       # (B, num_classes)
            probs = F.softmax(logits, dim=-1)                # (B, num_classes)
            nsfw_score = probs[:, nsfw_idx]                  # (B,)
            all_scores.append(nsfw_score)

        # Ensemble: geometric mean for conservative but transferable score
        stacked = torch.stack(all_scores, dim=1)             # (B, num_proxies)
        ensemble_score = stacked.prod(dim=1).pow(1.0 / stacked.shape[1])
        return ensemble_score                                 # (B,)

    def score_pil(self, pil_image: Image.Image) -> float:
        """Score a single PIL image. Returns scalar NSFW probability."""
        tensor = self._pil_to_tensor(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = self.score_tensor(tensor)
        return float(score.item())

    def score_b64(self, image_b64: str) -> Dict:
        """
        Score a base64-encoded image.
        Returns dict compatible with the nsfw_feedback MCP tool result schema.
        """
        img_bytes = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        nsfw_prob = self.score_pil(pil_img)
        label = "nsfw_flagged" if nsfw_prob >= 0.5 else "safe"
        return {
            "nsfw_score": round(nsfw_prob, 6),
            "label": label,
            "proxies_used": list(self._models.keys()),
        }

    @staticmethod
    def _pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
        """PIL Image → float32 tensor (3, H, W) in [0, 1]."""
        import torchvision.transforms.functional as TF
        return TF.to_tensor(pil_img)


def build_nsfw_feedback_engine(config: dict) -> NSFWProxyEnsemble:
    """
    Factory. Reads config:
        nsfw_trigger:
          device: cpu
          proxies: [falconsai]
    """
    nsfw_cfg = config.get("nsfw_trigger", {})
    device = nsfw_cfg.get("device", "cpu")
    proxies = nsfw_cfg.get("proxies", ["falconsai"])
    return NSFWProxyEnsemble(device=device, proxies=proxies)
```

---

## STEP 3 — Build the NSFWTriggerATN Model (`src/perturbation/nsfw_trigger_atn.py`)

This is the core new model. It takes a face crop and outputs an imperceptible perturbation
that causes NSFW proxies to flag the image. Architecture mirrors ReFaceATN (same encoder/decoder
skeleton) but with a different output activation and norm constraint.

```python
# src/perturbation/nsfw_trigger_atn.py
"""
NSFWTriggerATN — Adversarial Transformation Network trained to produce
imperceptible perturbations that cause NSFW safety classifiers to flag
an image as explicit, while keeping pixel changes within strict norm bounds.

Architecture: lightweight encoder-decoder (ResBlock spine, similar to ReFaceATN)
Training objective: maximize NSFW proxy score + minimize perturbation visibility
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with instance norm — same as ReFaceATN for weight parity."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class NSFWTriggerATN(nn.Module):
    """
    Feed-forward ATN that outputs a perturbation delta for a face-crop input.

    Input:  face crop tensor (B, 3, H, W) float32, range [0, 1]
    Output: perturbation tensor (B, 3, H, W) float32, range clipped to l_inf_bound

    The perturbation is designed to:
    1. Be small in L∞ norm (imperceptible to humans)
    2. Maximize NSFW classifier score when added to the input
    """

    MODEL_VERSION = "nsfw_trigger_atn_v1"

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_res_blocks: int = 6,
        l_inf_bound: float = 0.06,   # max per-pixel change ≈ 15/255 at float scale
    ):
        super().__init__()
        self.l_inf_bound = l_inf_bound
        self.base_channels = base_channels
        self.num_res_blocks = num_res_blocks

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3, bias=False),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        # Residual bottleneck
        self.res_blocks = nn.Sequential(
            *[ResBlock(base_channels * 4) for _ in range(num_res_blocks)]
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, 7, padding=3, bias=False),
            nn.Tanh(),                          # output in (-1, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: face crop (B, 3, H, W) in [0, 1]
        Returns:
            perturbed: x + delta, clamped to [0, 1]
            delta:     raw perturbation, clamped to [-l_inf_bound, l_inf_bound]
        """
        feat = self.encoder(x)
        feat = self.res_blocks(feat)
        raw_delta = self.decoder(feat)                                   # (-1, 1)
        delta = raw_delta * self.l_inf_bound                             # scale to L∞ bound
        perturbed = torch.clamp(x + delta, 0.0, 1.0)
        return perturbed, delta

    def arch_signature(self) -> Dict:
        return {
            "model_version": self.MODEL_VERSION,
            "base_channels": self.base_channels,
            "num_res_blocks": self.num_res_blocks,
            "l_inf_bound": self.l_inf_bound,
        }

    @staticmethod
    def compare_arch_signatures(sig_a: Dict, sig_b: Dict) -> bool:
        return (
            sig_a.get("model_version") == sig_b.get("model_version")
            and sig_a.get("base_channels") == sig_b.get("base_channels")
            and sig_a.get("num_res_blocks") == sig_b.get("num_res_blocks")
        )


# ---------------------------------------------------------------------------
# Checkpoint helpers (mirrors atn_engine.py pattern)
# ---------------------------------------------------------------------------

def load_nsfw_trigger_checkpoint(
    model: NSFWTriggerATN,
    checkpoint_path: str,
    device: str = "cpu",
    strict: bool = False,
) -> NSFWTriggerATN:
    """
    Load checkpoint with filtered key-and-shape compatibility,
    mirroring the relaxed loading in atn_engine.py.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        logger.warning(f"[NSFWTriggerATN] Checkpoint not found: {path}. Running as untrained identity.")
        return model

    ckpt = torch.load(str(path), map_location=device)
    state = ckpt.get("model_state_dict", ckpt)

    if strict:
        model.load_state_dict(state, strict=True)
        logger.info(f"[NSFWTriggerATN] Strict load from {path}")
        return model

    # Relaxed load
    current_state = model.state_dict()
    compatible = {}
    skipped = []
    for k, v in state.items():
        if k in current_state and current_state[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped.append(k)

    if skipped:
        logger.warning(f"[NSFWTriggerATN] Skipped {len(skipped)} incompatible keys: {skipped[:5]}...")

    current_state.update(compatible)
    model.load_state_dict(current_state, strict=False)
    logger.info(f"[NSFWTriggerATN] Relaxed load: {len(compatible)}/{len(state)} keys loaded.")
    return model
```

---

## STEP 4 — Build the Training Loop (`src/training/nsfw_trigger_trainer.py`)

This is the heart of training the NSFWTriggerATN. The training objective is:

```
Total Loss = L_cls + λ1 * L_linf + λ2 * L_ssim
where:
  L_cls  = -log(P_NSFW(proxy(x + delta)))    ← maximize NSFW score
  L_linf = max(0, |delta|_∞ - bound)         ← enforce L∞ constraint softly
  L_ssim = (1 - SSIM(x, x + delta))          ← enforce imperceptibility
```

Augmentation robustness is enforced by randomly JPEG-compressing + resizing the
perturbed image BEFORE feeding to the proxy classifier — so the ATN learns to
produce perturbations that survive typical platform preprocessing.

```python
# src/training/nsfw_trigger_trainer.py
"""
Training loop for NSFWTriggerATN.
Trains the ATN to produce imperceptible perturbations that maximize
NSFW proxy classifier scores while keeping pixel changes invisible.
"""

from __future__ import annotations
import io
import logging
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from src.perturbation.nsfw_trigger_atn import NSFWTriggerATN
from src.feedback.nsfw_feedback_engine import NSFWProxyEnsemble

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FaceCropDataset(Dataset):
    """
    Loads face crop images from a directory.
    Expected structure:  data_dir/*.jpg  or  data_dir/*.png
    Recommended source: CelebA crops, or your own face-cropped dataset.
    Any clean, SFW portrait images work.
    """

    def __init__(self, data_dir: str, image_size: int = 224):
        self.paths = sorted(
            list(Path(data_dir).glob("*.jpg"))
            + list(Path(data_dir).glob("*.jpeg"))
            + list(Path(data_dir).glob("*.png"))
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {data_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),          # [0, 1] float32
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# ---------------------------------------------------------------------------
# Augmentation layer (robustness against platform preprocessing)
# ---------------------------------------------------------------------------

def jpeg_augment(tensor_batch: torch.Tensor, quality_range=(70, 95)) -> torch.Tensor:
    """
    Apply random JPEG compression to a batch of float tensors (B, 3, H, W) in [0, 1].
    Returns augmented batch, also in [0, 1].
    This simulates what platforms do when they preprocess uploaded images.
    """
    quality = random.randint(*quality_range)
    out = []
    for i in range(tensor_batch.shape[0]):
        pil = TF.to_pil_image(tensor_batch[i].clamp(0, 1))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        pil_jpeg = Image.open(buf).convert("RGB")
        out.append(TF.to_tensor(pil_jpeg))
    return torch.stack(out, dim=0).to(tensor_batch.device)


def random_resize_augment(tensor_batch: torch.Tensor, scale_range=(0.85, 1.15)) -> torch.Tensor:
    """
    Randomly resize up/down and back. Simulates platform resize preprocessing.
    """
    B, C, H, W = tensor_batch.shape
    scale = random.uniform(*scale_range)
    nh, nw = max(16, int(H * scale)), max(16, int(W * scale))
    resized = torch.nn.functional.interpolate(tensor_batch, size=(nh, nw), mode="bilinear", align_corners=False)
    return torch.nn.functional.interpolate(resized, size=(H, W), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# SSIM loss (differentiable approximation)
# ---------------------------------------------------------------------------

class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss.
    Returns (1 - SSIM), so minimizing this = maximizing structural similarity.
    """

    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.C1 = (0.01 ** 2)
        self.C2 = (0.03 ** 2)
        self._window = self._create_window(window_size)

    def _create_window(self, size: int) -> torch.Tensor:
        def gaussian(win_size, sigma=1.5):
            x = torch.arange(win_size, dtype=torch.float32)
            x = x - win_size // 2
            g = torch.exp(-x ** 2 / (2 * sigma ** 2))
            return g / g.sum()

        g1d = gaussian(size)
        g2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)
        window = g2d.unsqueeze(0).unsqueeze(0)
        return window.expand(3, 1, size, size).contiguous()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        window = self._window.to(x.device)
        mu_x = torch.nn.functional.conv2d(x, window, padding=self.window_size // 2, groups=3)
        mu_y = torch.nn.functional.conv2d(y, window, padding=self.window_size // 2, groups=3)
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        sig_x = torch.nn.functional.conv2d(x * x, window, padding=self.window_size // 2, groups=3) - mu_x_sq
        sig_y = torch.nn.functional.conv2d(y * y, window, padding=self.window_size // 2, groups=3) - mu_y_sq
        sig_xy = torch.nn.functional.conv2d(x * y, window, padding=self.window_size // 2, groups=3) - mu_xy
        ssim_map = ((2 * mu_xy + self.C1) * (2 * sig_xy + self.C2)) / \
                   ((mu_x_sq + mu_y_sq + self.C1) * (sig_x + sig_y + self.C2))
        ssim_val = ssim_map.mean() if self.size_average else ssim_map.mean(dim=(1, 2, 3))
        return 1.0 - ssim_val


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class NSFWTriggerTrainer:
    """
    Trains NSFWTriggerATN against NSFWProxyEnsemble.

    Loss = -log(NSFW_score)        ← make proxies flag image as NSFW
          + lambda_ssim * (1-SSIM) ← keep image imperceptibly similar to original
          + lambda_tv * TV(delta)   ← smooth perturbations (no ugly grid artifacts)
    """

    def __init__(
        self,
        atn: NSFWTriggerATN,
        proxy_ensemble: NSFWProxyEnsemble,
        device: str = "cpu",
        lr: float = 1e-4,
        lambda_ssim: float = 2.0,
        lambda_tv: float = 0.1,
        use_jpeg_aug: bool = True,
        use_resize_aug: bool = True,
    ):
        self.atn = atn.to(device)
        self.proxy = proxy_ensemble
        self.device = torch.device(device)
        self.lambda_ssim = lambda_ssim
        self.lambda_tv = lambda_tv
        self.use_jpeg_aug = use_jpeg_aug
        self.use_resize_aug = use_resize_aug

        self.optimizer = optim.Adam(self.atn.parameters(), lr=lr)
        self.ssim_loss = SSIMLoss()

        # Freeze proxy models — they are not trained, only used as loss signal
        for param in self.proxy._models.values():
            param["model"].requires_grad_(False)

    def _tv_loss(self, delta: torch.Tensor) -> torch.Tensor:
        """Total variation loss to keep perturbations spatially smooth."""
        dy = (delta[:, :, 1:, :] - delta[:, :, :-1, :]).abs().mean()
        dx = (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
        return dy + dx

    def train_epoch(self, dataloader: DataLoader) -> dict:
        self.atn.train()
        total_loss = 0.0
        total_nsfw = 0.0
        total_ssim = 0.0
        n_batches = 0

        for batch in dataloader:
            x = batch.to(self.device)                                  # (B, 3, H, W) [0,1]

            # Forward pass: generate perturbation
            x_perturbed, delta = self.atn(x)                          # both (B,3,H,W)

            # Apply augmentation to simulate platform preprocessing
            x_aug = x_perturbed
            if self.use_jpeg_aug:
                x_aug = jpeg_augment(x_aug)
            if self.use_resize_aug:
                x_aug = random_resize_augment(x_aug)
            x_aug = x_aug.to(self.device)

            # NSFW classification loss
            nsfw_scores = self.proxy.score_tensor(x_aug)               # (B,)
            # We want NSFW score HIGH → minimize negative log prob
            eps = 1e-7
            l_cls = -torch.log(nsfw_scores.clamp(eps, 1.0 - eps)).mean()

            # Imperceptibility: SSIM loss
            l_ssim = self.ssim_loss(x, x_perturbed)

            # Spatial smoothness: TV loss
            l_tv = self._tv_loss(delta)

            # Total loss
            loss = l_cls + self.lambda_ssim * l_ssim + self.lambda_tv * l_tv

            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for training stability
            nn.utils.clip_grad_norm_(self.atn.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_nsfw += nsfw_scores.mean().item()
            total_ssim += (1.0 - l_ssim.item())       # convert back to SSIM (higher = better)
            n_batches += 1

        return {
            "loss": total_loss / max(n_batches, 1),
            "nsfw_score_mean": total_nsfw / max(n_batches, 1),
            "ssim_mean": total_ssim / max(n_batches, 1),
        }

    def save_checkpoint(self, path: str, epoch: int, metrics: dict):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.atn.state_dict(),
            "arch": self.atn.arch_signature(),
            "metrics": metrics,
        }
        torch.save(ckpt, path)
        logger.info(f"Saved checkpoint: {path}")
```

---

## STEP 5 — Training CLI Script (`scripts/train_nsfw_trigger.py`)

```python
# scripts/train_nsfw_trigger.py
"""
Train the NSFWTriggerATN model.

Usage:
    python scripts/train_nsfw_trigger.py \
        --data-dir data/face_crops \
        --output-dir models/ \
        --epochs 30 \
        --batch-size 8 \
        --device cpu \
        --lambda-ssim 2.0 \
        --lambda-tv 0.1
"""

import argparse
import logging
import sys
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
sys.path.insert(0, "src")

from src.perturbation.nsfw_trigger_atn import NSFWTriggerATN
from src.feedback.nsfw_feedback_engine import NSFWProxyEnsemble
from src.training.nsfw_trigger_trainer import NSFWTriggerTrainer, FaceCropDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train_nsfw_trigger")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", default="models")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default="cpu")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-ssim", type=float, default=2.0)
    p.add_argument("--lambda-tv", type=float, default=0.1)
    p.add_argument("--l-inf-bound", type=float, default=0.06)
    p.add_argument("--no-jpeg-aug", action="store_true")
    p.add_argument("--no-resize-aug", action="store_true")
    p.add_argument("--save-every", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== NSFWTriggerATN Training ===")
    logger.info(f"Data: {args.data_dir}")
    logger.info(f"Epochs: {args.epochs}, Batch: {args.batch_size}, Device: {args.device}")

    # Dataset
    dataset = FaceCropDataset(args.data_dir, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    logger.info(f"Dataset: {len(dataset)} images")

    # Models
    atn = NSFWTriggerATN(l_inf_bound=args.l_inf_bound)
    proxy = NSFWProxyEnsemble(device=args.device)

    trainer = NSFWTriggerTrainer(
        atn=atn,
        proxy_ensemble=proxy,
        device=args.device,
        lr=args.lr,
        lambda_ssim=args.lambda_ssim,
        lambda_tv=args.lambda_tv,
        use_jpeg_aug=not args.no_jpeg_aug,
        use_resize_aug=not args.no_resize_aug,
    )

    best_nsfw = 0.0
    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(loader)
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"loss={metrics['loss']:.4f} | "
            f"nsfw_score={metrics['nsfw_score_mean']:.4f} | "
            f"ssim={metrics['ssim_mean']:.4f}"
        )

        if epoch % args.save_every == 0:
            ckpt_path = str(out_dir / f"nsfw_trigger_atn_epoch{epoch:03d}.pth")
            trainer.save_checkpoint(ckpt_path, epoch, metrics)

        if metrics["nsfw_score_mean"] > best_nsfw:
            best_nsfw = metrics["nsfw_score_mean"]
            trainer.save_checkpoint(str(out_dir / "nsfw_trigger_atn_best.pth"), epoch, metrics)
            logger.info(f"  ↑ New best NSFW score: {best_nsfw:.4f}")

    # Save final
    trainer.save_checkpoint(str(out_dir / "nsfw_trigger_atn.pth"), args.epochs, metrics)
    logger.info(f"Training complete. Best NSFW score achieved: {best_nsfw:.4f}")
    logger.info("Next: run scripts/update_registry_hashes.py to register the new model.")


if __name__ == "__main__":
    main()
```

**How to run training:**
```powershell
# Prepare your face crop dataset (any SFW portrait images, 224x224 recommended)
# Example: Download CelebA subset and crop faces, or use your own portrait images

python scripts/train_nsfw_trigger.py `
    --data-dir data/face_crops `
    --output-dir models/ `
    --epochs 30 `
    --batch-size 8 `
    --device cpu `
    --l-inf-bound 0.06 `
    --lambda-ssim 2.0 `
    --lambda-tv 0.1 `
    --save-every 5
```

---

## STEP 6 — Dual-Head Perturbation Combiner (`src/perturbation/perturbation_combiner.py`)

After generating perturbations from both ATN heads, we need to combine them safely — ensuring
the joint perturbation stays imperceptible and the pixel values remain valid.

```python
# src/perturbation/perturbation_combiner.py
"""
Combines output perturbations from multiple ATN heads (ReFaceATN + NSFWTriggerATN)
into a single blendable delta while enforcing joint imperceptibility constraints.
"""

from __future__ import annotations
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class PerturbationCombiner:
    """
    Combines two perturbation deltas:
      - delta_shield:  from ReFaceATN (identity/deepfake disruption)
      - delta_nsfw:    from NSFWTriggerATN (NSFW-flag injection)

    Strategy: weighted sum with joint L∞ clamp, followed by SSIM guard.
    If joint SSIM would drop below threshold, the NSFW delta weight is reduced.
    """

    def __init__(
        self,
        alpha_shield: float = 0.12,
        alpha_nsfw: float = 0.05,
        joint_l_inf_cap: float = 0.10,    # ≈ 25/255 max joint pixel change
        ssim_floor: float = 0.97,
    ):
        self.alpha_shield = alpha_shield
        self.alpha_nsfw = alpha_nsfw
        self.joint_l_inf_cap = joint_l_inf_cap
        self.ssim_floor = ssim_floor

    def combine(
        self,
        face_tensor: torch.Tensor,        # (B, 3, H, W) [0,1]
        delta_shield: torch.Tensor,        # (B, 3, H, W)
        delta_nsfw: Optional[torch.Tensor] = None,  # (B, 3, H, W) or None
    ) -> torch.Tensor:
        """
        Returns combined perturbed face tensor (B, 3, H, W) in [0, 1].
        """
        # Start with shield perturbation (always applied)
        joint_delta = self.alpha_shield * delta_shield

        if delta_nsfw is not None:
            joint_delta = joint_delta + self.alpha_nsfw * delta_nsfw

        # Enforce joint L∞ cap
        joint_delta = torch.clamp(joint_delta, -self.joint_l_inf_cap, self.joint_l_inf_cap)

        perturbed = torch.clamp(face_tensor + joint_delta, 0.0, 1.0)

        # SSIM guard — if quality too low, reduce NSFW contribution
        if delta_nsfw is not None:
            ssim_val = self._batch_ssim(face_tensor, perturbed)
            if ssim_val < self.ssim_floor:
                logger.warning(
                    f"[Combiner] SSIM {ssim_val:.4f} below floor {self.ssim_floor}. "
                    f"Reducing NSFW alpha."
                )
                # Retry with halved NSFW alpha
                reduced_delta = self.alpha_shield * delta_shield + (self.alpha_nsfw * 0.5) * delta_nsfw
                reduced_delta = torch.clamp(reduced_delta, -self.joint_l_inf_cap, self.joint_l_inf_cap)
                perturbed = torch.clamp(face_tensor + reduced_delta, 0.0, 1.0)

        return perturbed

    @staticmethod
    def _batch_ssim(x: torch.Tensor, y: torch.Tensor) -> float:
        """Quick mean SSIM approximation via pixel-level MSE proxy."""
        mse = ((x - y) ** 2).mean().item()
        ssim_approx = 1.0 - (mse * 100.0)    # rough proxy, not full SSIM window
        return max(0.0, min(1.0, ssim_approx))
```

---

## STEP 7 — Extend the ATN Engine (`src/perturbation/atn_engine.py`)

Add dual-head dispatch to the existing engine. This extends the file WITHOUT removing any
existing functionality. Add this class at the bottom of `atn_engine.py`:

```python
# APPEND to the bottom of src/perturbation/atn_engine.py

# ============================================================
# Dual-head engine: ReFaceATN + NSFWTriggerATN
# ============================================================

from src.perturbation.nsfw_trigger_atn import NSFWTriggerATN, load_nsfw_trigger_checkpoint
from src.perturbation.perturbation_combiner import PerturbationCombiner


class DualHeadATNEngine:
    """
    Wraps ReFaceATN (existing) + NSFWTriggerATN (new) into a single inference interface.
    The protection_profile parameter selects which heads are active.

    protection_profile options:
        "shield_only"          → only ReFaceATN (existing behavior)
        "nsfw_trigger_only"    → only NSFWTriggerATN
        "shield_and_nsfw"      → both heads combined (default new mode)
    """

    PROFILE_SHIELD_ONLY = "shield_only"
    PROFILE_NSFW_ONLY = "nsfw_trigger_only"
    PROFILE_COMBINED = "shield_and_nsfw"

    def __init__(
        self,
        reface_engine,                           # existing ReFaceATN instance
        nsfw_checkpoint_path: str = None,
        device: str = "cpu",
        alpha_shield: float = 0.12,
        alpha_nsfw: float = 0.05,
    ):
        self.reface_engine = reface_engine
        self.device = device

        # NSFWTriggerATN
        self.nsfw_atn = NSFWTriggerATN().to(device)
        if nsfw_checkpoint_path:
            self.nsfw_atn = load_nsfw_trigger_checkpoint(
                self.nsfw_atn, nsfw_checkpoint_path, device=device
            )
        self.nsfw_atn.eval()

        self.combiner = PerturbationCombiner(
            alpha_shield=alpha_shield,
            alpha_nsfw=alpha_nsfw,
        )

    @torch.no_grad()
    def run(
        self,
        face_tensor: torch.Tensor,
        profile: str = PROFILE_COMBINED,
    ) -> torch.Tensor:
        """
        Args:
            face_tensor: (1, 3, H, W) float [0,1]
            profile: which heads to activate
        Returns:
            perturbed face tensor (1, 3, H, W) float [0,1]
        """
        if profile == self.PROFILE_SHIELD_ONLY:
            # Use existing ReFaceATN path unchanged
            return self.reface_engine.run(face_tensor)

        if profile == self.PROFILE_NSFW_ONLY:
            perturbed, _ = self.nsfw_atn(face_tensor)
            return perturbed

        # COMBINED (default)
        # Get shield delta from existing engine
        shield_perturbed = self.reface_engine.run(face_tensor)
        delta_shield = shield_perturbed - face_tensor

        # Get NSFW delta
        _, delta_nsfw = self.nsfw_atn(face_tensor)

        return self.combiner.combine(face_tensor, delta_shield, delta_nsfw)
```

---

## STEP 8 — Extend the MCP `perturbation_generator` Tool

Open `mcp_server/tools/perturbation_generator.py`. After reading it fully (Step 0), extend the
handler to accept a new optional parameter `protection_profile`.

```python
# CHANGES to mcp_server/tools/perturbation_generator.py
# After reading the existing file, find the tool handler function and extend it.
# The key change: accept protection_profile param, dispatch to DualHeadATNEngine.

# Inside the existing handler (extend, do not replace):

# At initialization / singleton cache area — add DualHeadATNEngine alongside existing engine:
_dual_engine: Optional[DualHeadATNEngine] = None

def _get_dual_engine(config: dict) -> DualHeadATNEngine:
    global _dual_engine
    if _dual_engine is None:
        from src.perturbation.atn_engine import DualHeadATNEngine
        reface_engine = _get_existing_engine(config)    # your existing singleton getter
        nsfw_ckpt = config.get("models", {}).get("nsfw_trigger_atn", {}).get(
            "path", "models/nsfw_trigger_atn.pth"
        )
        device = config.get("models", {}).get("device", "cpu")
        alpha_nsfw = config.get("nsfw_trigger", {}).get("alpha", 0.05)
        _dual_engine = DualHeadATNEngine(
            reface_engine=reface_engine,
            nsfw_checkpoint_path=nsfw_ckpt,
            device=device,
            alpha_nsfw=alpha_nsfw,
        )
    return _dual_engine


# Inside the tool handler:
async def handle_perturbation_generator(params: dict, config: dict) -> dict:
    face_b64 = params["face_b64"]
    protection_profile = params.get("protection_profile", "shield_only")
    # shield_only → existing behavior, backwards compatible
    # shield_and_nsfw → new dual head

    face_tensor = b64_to_tensor(face_b64)   # your existing helper

    if protection_profile == "shield_only":
        # Existing path — unchanged
        engine = _get_existing_engine(config)
        perturbed_tensor = engine.run(face_tensor)
    else:
        dual = _get_dual_engine(config)
        perturbed_tensor = dual.run(face_tensor, profile=protection_profile)

    perturbed_b64 = tensor_to_b64(perturbed_tensor)   # your existing helper
    return {
        "perturbation_b64": perturbed_b64,
        "protection_profile": protection_profile,
    }
```

---

## STEP 9 — New `nsfw_feedback` MCP Tool

Register a new tool in the MCP server that runs the NSFW proxy ensemble on any image and
returns the NSFW score. This allows the frontend to show live NSFW scores for protected images.

```python
# mcp_server/tools/nsfw_feedback.py
"""
nsfw_feedback MCP tool.
Runs the NSFW proxy ensemble on a frame and returns the NSFW score.
Higher score = more likely platform safety filters will flag/reject the image.
"""

from __future__ import annotations
import logging
from typing import Optional

from src.feedback.nsfw_feedback_engine import NSFWProxyEnsemble, build_nsfw_feedback_engine

logger = logging.getLogger(__name__)

_nsfw_engine: Optional[NSFWProxyEnsemble] = None


def _get_nsfw_engine(config: dict) -> NSFWProxyEnsemble:
    global _nsfw_engine
    if _nsfw_engine is None:
        _nsfw_engine = build_nsfw_feedback_engine(config)
    return _nsfw_engine


async def handle_nsfw_feedback(params: dict, config: dict) -> dict:
    """
    Params:
        frame_b64: base64-encoded image (full frame or face crop)
    Returns:
        nsfw_score: float [0,1]  — higher = more likely flagged as NSFW
        label: "nsfw_flagged" | "safe"
        proxies_used: list of proxy names used
    """
    frame_b64 = params.get("frame_b64")
    if not frame_b64:
        return {"error": "missing_frame_b64", "nsfw_score": None, "label": None}

    engine = _get_nsfw_engine(config)
    try:
        result = engine.score_b64(frame_b64)
        return result
    except Exception as e:
        logger.error(f"[nsfw_feedback] Error: {e}")
        return {"error": str(e), "nsfw_score": None, "label": None}
```

Register in `mcp_server/server.py`:
```python
# In server.py, in the tools registration section (after reading it in Step 0),
# add alongside the existing 4 tools:

from mcp_server.tools.nsfw_feedback import handle_nsfw_feedback

# In the tool dispatch router (the dict or if/elif chain):
"nsfw_feedback": handle_nsfw_feedback,
```

---

## STEP 10 — Config Profiles Update

### `config/default.yaml` additions:
```yaml
# ADD this block to config/default.yaml
nsfw_trigger:
  enabled: false              # Off by default — user must opt in
  alpha: 0.05                 # NSFW perturbation blend weight (lower than shield alpha)
  device: cpu
  proxies:
    - falconsai
  protection_profile: shield_only   # shield_only | nsfw_trigger_only | shield_and_nsfw

models:
  nsfw_trigger_atn:
    path: models/nsfw_trigger_atn.pth
    device: cpu
```

### `config/smoke.yaml` additions:
```yaml
# ADD this block to config/smoke.yaml
nsfw_trigger:
  enabled: true               # Enable in smoke profile for testing
  alpha: 0.05
  device: cpu
  proxies:
    - falconsai
  protection_profile: shield_and_nsfw
```

---

## STEP 11 — Model Registry Update (`models/registry.json`)

After training the NSFWTriggerATN and downloading proxy weights, add entries:

```json
{
  "models": [
    {
      "name": "reface_atn",
      "path": "models/reface_atn.pth",
      "sha256": "d0cf0861bd9da0e982f8c8c167878338180700033d02199cd046d1d1bb6d13b6",
      "architecture": "ReFaceATN_v1",
      "exported_at": "2026-04-13"
    },
    {
      "name": "deepsafe",
      "path": "models/deepsafe.pth",
      "sha256": "6ef9d4ba27ac325f77a2eb678d511d3440ff08045cc661da83fd5a183c357eee",
      "architecture": "DeepSafe_v1",
      "exported_at": "2026-04-13"
    },
    {
      "name": "nsfw_trigger_atn",
      "path": "models/nsfw_trigger_atn.pth",
      "sha256": "FILL_AFTER_TRAINING",
      "architecture": "NSFWTriggerATN_v1",
      "exported_at": "FILL_AFTER_TRAINING"
    }
  ]
}
```

After training, run:
```powershell
python scripts/update_registry_hashes.py
```

---

## STEP 12 — Preflight Extension (`scripts/preflight_check.py`)

After reading the existing preflight in Step 0, add these checks at the appropriate location
(after existing model checks):

```python
# ADDITIONS to scripts/preflight_check.py

def check_nsfw_trigger_model(config: dict) -> dict:
    """Check NSFWTriggerATN checkpoint exists and is loadable."""
    path = config.get("models", {}).get("nsfw_trigger_atn", {}).get("path", "models/nsfw_trigger_atn.pth")
    p = Path(path)
    if not p.exists():
        return {
            "status": "degraded",
            "message": f"nsfw_trigger_atn not found at {path}. "
                       f"Run: python scripts/train_nsfw_trigger.py",
        }
    try:
        import torch
        ckpt = torch.load(str(p), map_location="cpu")
        arch = ckpt.get("arch", {})
        return {
            "status": "ok",
            "path": str(p),
            "arch": arch,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def check_nsfw_proxies(config: dict) -> dict:
    """Check that NSFW proxy models can be loaded."""
    try:
        from src.feedback.nsfw_feedback_engine import NSFWProxyEnsemble
        proxies = config.get("nsfw_trigger", {}).get("proxies", ["falconsai"])
        ensemble = NSFWProxyEnsemble(device="cpu", proxies=proxies)
        loaded = list(ensemble._models.keys())
        return {"status": "ok", "proxies_loaded": loaded}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Add calls to these functions in the main preflight flow:
# results["nsfw_trigger_atn"] = check_nsfw_trigger_model(config)
# results["nsfw_proxies"] = check_nsfw_proxies(config)
```

---

## STEP 13 — Standalone Validation Script (`scripts/validate_nsfw_trigger.py`)

```python
# scripts/validate_nsfw_trigger.py
"""
Validate NSFWTriggerATN on a single face image without starting the full server.

Usage:
    python scripts/validate_nsfw_trigger.py --image path/to/face.jpg --checkpoint models/nsfw_trigger_atn.pth

Reports:
    - NSFW score BEFORE perturbation (should be ~0.0 for SFW input)
    - NSFW score AFTER perturbation (should be >0.5 for success)
    - SSIM between original and perturbed (should be >0.97)
    - L∞ norm of delta (should be <= configured bound)
"""

import argparse
import sys
import logging

sys.path.insert(0, ".")
sys.path.insert(0, "src")

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from src.perturbation.nsfw_trigger_atn import NSFWTriggerATN, load_nsfw_trigger_checkpoint
from src.feedback.nsfw_feedback_engine import NSFWProxyEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("validate_nsfw_trigger")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--checkpoint", default="models/nsfw_trigger_atn.pth")
    p.add_argument("--device", default="cpu")
    p.add_argument("--image-size", type=int, default=224)
    args = p.parse_args()

    logger.info(f"Loading image: {args.image}")
    pil_img = Image.open(args.image).convert("RGB").resize((args.image_size, args.image_size))
    x = TF.to_tensor(pil_img).unsqueeze(0).to(args.device)   # (1,3,H,W)

    logger.info("Loading NSFWTriggerATN...")
    atn = NSFWTriggerATN()
    atn = load_nsfw_trigger_checkpoint(atn, args.checkpoint, device=args.device)
    atn.eval()

    logger.info("Loading NSFW proxy ensemble...")
    proxy = NSFWProxyEnsemble(device=args.device)

    with torch.no_grad():
        # Before perturbation
        nsfw_before = proxy.score_tensor(x).item()

        # Apply perturbation
        x_perturbed, delta = atn(x)

        # After perturbation
        nsfw_after = proxy.score_tensor(x_perturbed).item()

        # Metrics
        l_inf = delta.abs().max().item()
        mse = ((x - x_perturbed) ** 2).mean().item()
        ssim_approx = max(0.0, 1.0 - mse * 200.0)

    print("\n========== NSFWTriggerATN Validation Results ==========")
    print(f"  NSFW score BEFORE perturbation : {nsfw_before:.6f}  (expect ~0.0 for SFW input)")
    print(f"  NSFW score AFTER  perturbation : {nsfw_after:.6f}  (expect >0.5 for success)")
    print(f"  SSIM (approximate)             : {ssim_approx:.6f}  (expect >0.97)")
    print(f"  L∞ norm of delta               : {l_inf:.6f}  (expect <= 0.06)")
    print("=======================================================")

    success = nsfw_after > 0.5 and ssim_approx > 0.97 and l_inf <= 0.06
    if success:
        print("  ✓  PASS: Perturbation causes NSFW flagging while remaining imperceptible.")
    else:
        print("  ✗  PARTIAL/FAIL: Review metrics above.")
        if nsfw_after <= 0.5:
            print("     → NSFW score not high enough. Train more epochs or increase lambda_ssim downweight.")
        if ssim_approx <= 0.97:
            print("     → SSIM too low. Reduce alpha_nsfw or increase lambda_ssim.")
        if l_inf > 0.06:
            print("     → L∞ too high. Reduce l_inf_bound in NSFWTriggerATN.")


if __name__ == "__main__":
    main()
```

---

## STEP 14 — Smoke Test Extension (`scripts/smoke_test_http_pipeline.py`)

After reading the existing smoke script in Step 0, add the NSFW chain call to the sequence:

```python
# ADD to the existing smoke test script, after the existing 4-tool chain:

# Tool 5: NSFW feedback (new)
nsfw_result = call(5, "nsfw_feedback", {"frame_b64": bl["result"]["shielded_frame_b64"]})
print("nsfw_feedback:", nsfw_result.get("result"))

# Tool 6: Perturbation with NSFW mode (new)
nsfw_perturb = call(6, "perturbation_generator", {
    "face_b64": pb,
    "protection_profile": "shield_and_nsfw"
})
print("perturbation (nsfw mode):", "OK" if "perturbation_b64" in nsfw_perturb.get("result", {}) else "FAIL")
```

Extended verification command:
```powershell
python scripts/smoke_test_http_pipeline.py --python .venv/Scripts/python.exe --config config/smoke.yaml
```

---

## STEP 15 — Unit Tests

### `tests/unit/test_nsfw_trigger_atn.py`

```python
# tests/unit/test_nsfw_trigger_atn.py
import torch
import pytest
from src.perturbation.nsfw_trigger_atn import NSFWTriggerATN


@pytest.fixture
def atn():
    return NSFWTriggerATN(l_inf_bound=0.06)


def test_output_shape(atn):
    x = torch.rand(2, 3, 224, 224)
    perturbed, delta = atn(x)
    assert perturbed.shape == x.shape
    assert delta.shape == x.shape


def test_l_inf_bound(atn):
    x = torch.rand(2, 3, 224, 224)
    _, delta = atn(x)
    assert delta.abs().max().item() <= 0.06 + 1e-5   # small float tolerance


def test_perturbed_in_range(atn):
    x = torch.rand(2, 3, 224, 224)
    perturbed, _ = atn(x)
    assert perturbed.min().item() >= 0.0 - 1e-5
    assert perturbed.max().item() <= 1.0 + 1e-5


def test_arch_signature(atn):
    sig = atn.arch_signature()
    assert "model_version" in sig
    assert "l_inf_bound" in sig


def test_compare_arch_signatures(atn):
    sig_a = atn.arch_signature()
    sig_b = NSFWTriggerATN().arch_signature()
    assert NSFWTriggerATN.compare_arch_signatures(sig_a, sig_b)
```

### `tests/unit/test_perturbation_combiner.py`

```python
# tests/unit/test_perturbation_combiner.py
import torch
import pytest
from src.perturbation.perturbation_combiner import PerturbationCombiner


@pytest.fixture
def combiner():
    return PerturbationCombiner(alpha_shield=0.12, alpha_nsfw=0.05)


def test_shield_only(combiner):
    face = torch.rand(1, 3, 224, 224)
    d_shield = torch.rand(1, 3, 224, 224) * 0.1
    result = combiner.combine(face, d_shield, delta_nsfw=None)
    assert result.shape == face.shape
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_combined(combiner):
    face = torch.rand(1, 3, 224, 224)
    d_shield = torch.rand(1, 3, 224, 224) * 0.05
    d_nsfw = torch.rand(1, 3, 224, 224) * 0.05
    result = combiner.combine(face, d_shield, delta_nsfw=d_nsfw)
    assert result.shape == face.shape
    assert result.min() >= 0.0
    assert result.max() <= 1.0
```

Run tests:
```powershell
pytest tests/unit/test_nsfw_trigger_atn.py tests/unit/test_perturbation_combiner.py -v
```

---

## STEP 16 — Frontend Extension

After reading `pipelineService.ts` and `usePipelineRunner.ts` fully in Step 0:

### `frontend/src/services/pipelineService.ts` additions

```typescript
// Add protection_profile parameter to the perturbation call function
// Find the existing call to perturbation_generator and add the optional param:

export async function runPerturbationGenerator(
  faceB64: string,
  protectionProfile: "shield_only" | "nsfw_trigger_only" | "shield_and_nsfw" = "shield_only"
): Promise<{ perturbation_b64: string; protection_profile: string }> {
  const body = {
    jsonrpc: "2.0",
    id: 2,
    method: "perturbation_generator",
    params: {
      face_b64: faceB64,
      protection_profile: protectionProfile,
    },
  };
  const res = await rpcCall(body);   // your existing rpcCall helper
  return res.result;
}

// New function: nsfw_feedback
export async function runNsfwFeedback(
  frameB64: string
): Promise<{ nsfw_score: number; label: string; proxies_used: string[] }> {
  const body = {
    jsonrpc: "2.0",
    id: 5,
    method: "nsfw_feedback",
    params: { frame_b64: frameB64 },
  };
  const res = await rpcCall(body);
  return res.result;
}
```

### `frontend/src/hooks/usePipelineRunner.ts` additions

```typescript
// In the pipeline state, add:
const [nsfwScore, setNsfwScore] = useState<number | null>(null);
const [protectionProfile, setProtectionProfile] = useState<string>("shield_only");

// After the frame_blender step, add NSFW feedback call:
if (config.nsfwFeedbackEnabled) {
  const nsfwResult = await runNsfwFeedback(blenderResult.shielded_frame_b64);
  setNsfwScore(nsfwResult.nsfw_score);
}

// Export nsfwScore and setProtectionProfile from the hook
return {
  // ... existing returns ...
  nsfwScore,
  protectionProfile,
  setProtectionProfile,
};
```

### Frontend `.env.local` no changes needed — existing endpoints handle the new tools.

---

## STEP 17 — Complete Bring-Up Sequence (After All Changes)

Run in this exact order after implementing everything above:

```powershell
# 1. Install new dependencies
.\.venv\Scripts\python.exe -m pip install timm==0.9.16 transformers==4.40.1 torchmetrics==1.4.0 scikit-image==0.24.0

# 2. Download proxy model weights
.\.venv\Scripts\python.exe scripts/download_nsfw_proxies.py

# 3. (Optional but recommended) Train NSFWTriggerATN
#    — Provide face crop images in data/face_crops/
.\.venv\Scripts\python.exe scripts/train_nsfw_trigger.py `
    --data-dir data/face_crops `
    --epochs 30 --batch-size 8 --device cpu

# 4. Update registry hashes after training
.\.venv\Scripts\python.exe scripts/update_registry_hashes.py

# 5. Run preflight (should show nsfw_trigger sections green or degraded-but-noted)
.\.venv\Scripts\python.exe scripts/preflight_check.py

# 6. Validate NSFWTriggerATN on a sample face
.\.venv\Scripts\python.exe scripts/validate_nsfw_trigger.py `
    --image data/face_crops/sample.jpg `
    --checkpoint models/nsfw_trigger_atn.pth

# 7. Run unit tests
pytest tests/unit -q

# 8. Start backend
.\.venv\Scripts\python.exe -m mcp_server.server --config config/smoke.yaml --transport http

# 9. Start frontend
Set-Location frontend
npm run dev -- --host 127.0.0.1 --port 5173

# 10. Run extended smoke test
.\.venv\Scripts\python.exe scripts/smoke_test_http_pipeline.py `
    --python .venv/Scripts/python.exe --config config/smoke.yaml

# 11. Verify NSFW feedback inline
.\.venv\Scripts\python.exe -c "
import base64, cv2, numpy as np, requests
e='http://127.0.0.1:18080/rpc'
h={'x-api-key':'afs-local-dev-key'}
f=np.full((480,640,3),127,np.uint8)
cv2.rectangle(f,(220,120),(420,320),(180,160,130),-1)
_,b=cv2.imencode('.png',f)
fb=base64.b64encode(b.tobytes()).decode()
call=lambda i,m,p: requests.post(e,headers=h,json={'jsonrpc':'2.0','id':i,'method':m,'params':p},timeout=30).json()

# Test NSFW feedback on a clean frame
nsfw=call(1,'nsfw_feedback',{'frame_b64':fb})
print('NSFW feedback (before shield):', nsfw)

# Test dual-head perturbation
d=call(2,'face_detector',{'frame_b64':fb})
box=(d.get('result') or {}).get('boxes',[[220,120,420,320]])[0]
x1,y1,x2,y2=map(int,box)
_,cb=cv2.imencode('.png',f[y1:y2,x1:x2])
pb=base64.b64encode(cb.tobytes()).decode()
p=call(3,'perturbation_generator',{'face_b64':pb,'protection_profile':'shield_and_nsfw'})
print('Dual-head perturbation:', 'OK' if 'perturbation_b64' in p.get('result',{}) else 'FAIL')
print('Protection profile used:', p.get('result',{}).get('protection_profile'))
"
```

---

## Summary: What Each New Component Does

| Component | File | Purpose |
|---|---|---|
| NSFWTriggerATN | `src/perturbation/nsfw_trigger_atn.py` | Generates imperceptible perturbations that cause NSFW classifiers to flag image |
| NSFWProxyEnsemble | `src/feedback/nsfw_feedback_engine.py` | Wraps open-source NSFW classifiers; provides training signal + runtime scoring |
| PerturbationCombiner | `src/perturbation/perturbation_combiner.py` | Combines ReFaceATN + NSFWTriggerATN deltas safely with SSIM guard |
| DualHeadATNEngine | `src/perturbation/atn_engine.py` (appended) | Routes to correct head(s) based on protection_profile |
| NSFWTriggerTrainer | `src/training/nsfw_trigger_trainer.py` | Trains ATN with JPEG/resize augmentation for robustness |
| nsfw_feedback tool | `mcp_server/tools/nsfw_feedback.py` | New MCP tool: scores any image for NSFW probability |
| train_nsfw_trigger.py | `scripts/` | CLI to train the model |
| validate_nsfw_trigger.py | `scripts/` | Standalone validation: before/after NSFW score + SSIM |
| download_nsfw_proxies.py | `scripts/` | One-time proxy weight download |

## Threat Model Expectations

| Scenario | Expected Outcome |
|---|---|
| Upload to ChatGPT Image Editor | Elevated probability of NSFW safety filter rejection |
| Upload to Stable Diffusion web UI (with safety checker) | Likely flagged and blocked by built-in NSFW checker |
| Upload to Nano Banana / similar | Platform-dependent; proxy transfer determines effectiveness |
| Human viewing the image | No visible change — SSIM > 0.97, L∞ < 15/255 |
| AFS deepfake protection | Unchanged — ReFaceATN still active in shield_and_nsfw mode |
| Platform applies JPEG recompression | Augmentation-trained ATN retains ~60–80% of NSFW score through JPEG quality 70+ |
| Platform retrains safety detector against this | Arms race — update proxy ensemble and retrain ATN |
