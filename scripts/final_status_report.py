import json
from pathlib import Path

BASE = Path("D:/deepfake_detection")


def count_images(d: Path) -> int:
    if not d.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return len([f for f in d.rglob("*") if f.suffix.lower() in exts])


items = [
    ("data/real images", count_images(BASE / "data" / "real"), ">= 5000"),
    ("data/fake images", count_images(BASE / "data" / "fake"), ">= 1000"),
    ("data/atn_train images", count_images(BASE / "data" / "atn_train"), ">= 3500"),
    ("data/atn_val images", count_images(BASE / "data" / "atn_val"), ">= 900"),
    ("models/reface_atn.pth exists", (BASE / "models" / "reface_atn.pth").exists(), True),
    ("models/deepsafe.pth exists", (BASE / "models" / "deepsafe.pth").exists(), True),
    ("registry.json updated", (BASE / "models" / "registry.json").exists(), True),
    ("01_train_atn.ipynb exists", (BASE / "notebooks" / "01_train_atn.ipynb").exists(), True),
    ("02_setup_deepsafe.ipynb exists", (BASE / "notebooks" / "02_setup_deepsafe.ipynb").exists(), True),
    ("03_inference_test.ipynb exists", (BASE / "notebooks" / "03_inference_test.ipynb").exists(), True),
]

print(f"\n{'Item':<40} {'Value':<14} {'Target':<10} {'Status':>6}")
print("-" * 76)

all_ok = True
for name, value, target in items:
    if isinstance(target, bool):
        ok = value == target
        display_val = "Yes" if value else "No"
        display_target = "Present"
    else:
        threshold = int(target.replace(">=", "").strip())
        ok = int(value) >= threshold
        display_val = str(value)
        display_target = target

    all_ok = all_ok and ok
    status = "PASS" if ok else "FAIL"
    print(f"{name:<40} {display_val:<14} {display_target:<10} {status:>6}")

print("\nALL DONE - Models ready for AFS pipeline." if all_ok else "\nSome items are incomplete. Check FAIL rows above.")

# Optional: print latest benchmark artifact if available.
artifact = BASE / "validation" / "benchmark_results.json"
if artifact.exists():
    try:
        with artifact.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        metrics = payload.get("metrics", {})
        print("\nLatest benchmark artifact:")
        print(f"  SSIM: {metrics.get('ssim_shielded_vs_original', 'n/a')}")
        print(f"  Fake prob: {metrics.get('fake_prob_shielded', 'n/a')}")
        print(f"  Score drop: {metrics.get('score_drop_orig_minus_shield', 'n/a')}")
        print(f"  P95 latency ms: {metrics.get('p95_total_latency_ms', 'n/a')}")
    except Exception as ex:
        print(f"\nCould not parse benchmark artifact: {ex}")
