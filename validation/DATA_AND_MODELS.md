# Data and Models Reference

This document is a quick-reference inventory for datasets and model artifacts used by this project.

## Scope

- Workspace root: D:/deepfake_detection
- Primary data root: D:/deepfake_detection/data
- Primary model root: D:/deepfake_detection/models

## Dataset Inventory

### Active local datasets

| Name | Type | Local Path | Current Count | Source | Method | Intended Use |
|---|---|---|---:|---|---|---|
| Real Faces (LFW-derived) | Real | D:/deepfake_detection/data/real | 13233 | sklearn.datasets.fetch_lfw_people (LFW) | Programmatic fallback download | Base real-face corpus |
| ATN Train Split | Real split | D:/deepfake_detection/data/atn_train | 4000 | Derived from data/real | Deterministic split (seed=42) | Notebook 01 training |
| ATN Val Split | Real split | D:/deepfake_detection/data/atn_val | 1000 | Derived from data/real | Deterministic split (seed=42) | Notebook 01 validation |
| Fake Faces (Augmented Stand-in) | Synthetic fake fallback | D:/deepfake_detection/data/fake | 1000 | Locally generated from data/real | Heavy augmentation fallback | Pipeline testing only |

### Preferred dataset sources used by scripts

| Dataset | URL | Target Path | Notes |
|---|---|---|---|
| CelebA-HQ Resized 256x256 | https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256 | D:/deepfake_detection/data/real | Preferred real-data source |
| 140k Real and Fake Faces | https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces | D:/deepfake_detection/data/fake | Copy fake subset only |
| FFHQ (fallback) | https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq | D:/deepfake_detection/data/real | Manual fallback |
| DFDC sample (fallback) | https://www.kaggle.com/datasets/humananalog/deepfake-detection-challenge-sample | D:/deepfake_detection/data/fake | Manual fallback |
| DFDC official portal | https://dfdc.ai/ | D:/deepfake_detection/data/fake | Manual fallback |

### Dataset quality warning

- The current fake set in D:/deepfake_detection/data/fake is an augmented-real fallback.
- It should be replaced with real deepfake/generated samples for production-quality training.

## Model Inventory

### Core outputs expected from notebooks

| Model Name | File | Local Path | Produced By | Purpose |
|---|---|---|---|---|
| reface_atn | reface_atn.pth | D:/deepfake_detection/models/reface_atn.pth | notebooks/01_train_atn.ipynb | Adversarial perturbation generator |
| deepsafe | deepsafe.pth | D:/deepfake_detection/models/deepsafe.pth | notebooks/02_setup_deepsafe.ipynb | Deepfake detector/feedback |

### External pretrained weights present in workspace

| Weight | Local Path | Source Repo | Source URL |
|---|---|---|---|
| fc_weights.pth | D:/deepfake_detection/gitrepos/UniversalFakeDetect/pretrained_weights/fc_weights.pth | Yuheng-Li/UniversalFakeDetect | https://github.com/Yuheng-Li/UniversalFakeDetect |
| NPR.pth | D:/deepfake_detection/gitrepos/NPR-DeepfakeDetection/checkpoints/NPR.pth | chuangchuangtan/NPR-DeepfakeDetection | https://github.com/chuangchuangtan/NPR-DeepfakeDetection |
| model_epoch_last_3090.pth | D:/deepfake_detection/gitrepos/CrossEfficientViT/checkpoints/model_epoch_last_3090.pth | Cross-path compatibility placement | Filename originally observed in NPR-DeepfakeDetection |

## Registry and Hashes

- Registry file: D:/deepfake_detection/models/registry.json
- Script that updates hashes: scripts/run_phase2.py
- Hashes to be updated after notebook execution:
  - reface_atn
  - deepsafe

## Phase 2 scripts

- scripts/preflight_check.py
- scripts/download_data.py
- scripts/run_phase2.py

## Notes for manual runs

1. Run preflight first.
2. If data checks fail, run data download/acquisition.
3. Run notebook execution pipeline.
4. Confirm model files exist and registry hashes are updated.
