import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.update_registry_hashes import validate_registry
from src.detection.face_detector import check_mediapipe

BASE = Path("D:/deepfake_detection")
REAL_DIR = BASE / "data/real"
FAKE_DIR = BASE / "data/fake"
TRAIN_DIR = BASE / "data/atn_train"
VAL_DIR = BASE / "data/atn_val"
MODELS_DIR = BASE / "models"
WEIGHTS = {
    "fc_weights.pth": BASE / "gitrepos/UniversalFakeDetect/pretrained_weights/fc_weights.pth",
    "NPR.pth": BASE / "gitrepos/NPR-DeepfakeDetection/checkpoints/NPR.pth",
    "model_epoch_last_3090.pth": BASE / "gitrepos/CrossEfficientViT/checkpoints/model_epoch_last_3090.pth",
}


def count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return len(
        [
            f
            for f in directory.rglob("*")
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
    )


def run_preflight() -> bool:
    checks = []

    # Data checks
    checks.append(("real/ exists", REAL_DIR.exists()))
    checks.append(("real/ has >=5000 imgs", count_images(REAL_DIR) >= 5000))
    checks.append(("fake/ exists", FAKE_DIR.exists()))
    checks.append(("fake/ has >=1000 imgs", count_images(FAKE_DIR) >= 1000))
    checks.append(("atn_train/ has ~4000", count_images(TRAIN_DIR) >= 3500))
    checks.append(("atn_val/ has ~1000", count_images(VAL_DIR) >= 900))

    # Weight checks
    for name, path in WEIGHTS.items():
        checks.append((f"{name} present", path.exists()))

    # Model output dirs
    checks.append(("models/ dir exists", MODELS_DIR.exists()))
    checks.append(("models/checkpoints/ exists", (MODELS_DIR / "checkpoints").exists()))
    registry_path = MODELS_DIR / "registry.json"
    checks.append(("registry.json exists", registry_path.exists()))
    checks.append(("mediapipe API available", check_mediapipe()))

    registry_violations = validate_registry(registry_path)
    checks.append(("registry schema valid", len(registry_violations) == 0))

    print(f"\n{'Check':<40} {'Status':>8}")
    print("-" * 50)
    all_pass = True
    for name, result in checks:
        icon = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"{name:<40} {icon:>8}")

    print("\n" + ("ALL CHECKS PASSED" if all_pass else "BLOCKERS FOUND - resolve before continuing"))

    if registry_violations:
        print("\n[Registry Violations]")
        for issue in registry_violations:
            print(f"- {issue}")

    print(
        f"\n[Counts] real={count_images(REAL_DIR)}  fake={count_images(FAKE_DIR)}  "
        f"atn_train={count_images(TRAIN_DIR)}  atn_val={count_images(VAL_DIR)}"
    )

    return all_pass


if __name__ == "__main__":
    ok = run_preflight()
    raise SystemExit(0 if ok else 1)