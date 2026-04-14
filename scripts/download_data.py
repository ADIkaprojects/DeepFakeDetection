import glob
import random
import shutil
import subprocess
import sys
from pathlib import Path


BASE = Path("D:/deepfake_detection")
REAL_DIR = BASE / "data/real"
FAKE_DIR = BASE / "data/fake"
TRAIN_DIR = BASE / "data/atn_train"
VAL_DIR = BASE / "data/atn_val"
TMP_CELEBA = BASE / "data/tmp_celebahq"
TMP_FAKE = BASE / "data/tmp_fake"
TMP_LFW = BASE / "data/tmp_lfw"


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


def ensure_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)


def has_kaggle_token() -> bool:
    token_path = Path.home() / ".kaggle" / "kaggle.json"
    return token_path.exists()


def method_2a_kaggle_celebahq() -> bool:
    print("[2A] Trying Method 1: CelebA-HQ via Kaggle")
    if not has_kaggle_token():
        print("[2A] Kaggle token missing.")
        print("URL: https://www.kaggle.com/settings/account")
        print("Save token to: C:/Users/<your-user>/.kaggle/kaggle.json")
        input("Press Enter after placing kaggle.json, or Ctrl+C to stop...")
        return False

    ensure_dir(REAL_DIR)
    ensure_dir(TMP_CELEBA)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            "badasstechie/celebahq-resized-256x256",
            "-p",
            str(TMP_CELEBA),
            "--unzip",
        ],
        check=True,
    )

    copied = 0
    for pattern in ("**/*.jpg", "**/*.png"):
        for file_path in glob.glob(str(TMP_CELEBA / pattern), recursive=True):
            src = Path(file_path)
            dst = REAL_DIR / src.name
            if not dst.exists():
                shutil.copy(src, dst)
                copied += 1

    print(f"[2A] Method 1 succeeded. Copied {copied} images. real/ now has {count_images(REAL_DIR)} images")
    return count_images(REAL_DIR) >= 5000


def method_2a_lfw() -> bool:
    print("[2A] Trying Method 3: LFW fallback")
    ensure_dir(REAL_DIR)
    ensure_dir(TMP_LFW)

    try:
        import cv2
        import numpy as np
        from sklearn.datasets import fetch_lfw_people
    except ImportError as exc:
        print(f"[2A] Missing dependency for LFW fallback: {exc}")
        print("Install and retry: pip install scikit-learn opencv-python")
        return False

    lfw = fetch_lfw_people(
        min_faces_per_person=1,
        resize=1.0,
        data_home=str(TMP_LFW),
        color=True,
        download_if_missing=True,
    )

    saved = 0
    for i, img in enumerate(lfw.images):
        img_uint8 = (img * 255).astype(np.uint8)
        out_path = REAL_DIR / f"lfw_{i:05d}.jpg"
        if not out_path.exists():
            cv2.imwrite(str(out_path), cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
            saved += 1

    print(f"[2A-LFW] Saved {saved} new images to real/. real/ now has {count_images(REAL_DIR)} images")
    return count_images(REAL_DIR) >= 5000


def step_2a_real_data() -> bool:
    current = count_images(REAL_DIR)
    if current >= 5000:
        print(f"[2A] real/ already has {current} images. Skipped.")
        return True

    try:
        if method_2a_kaggle_celebahq():
            print("[2A] Success via Method 1 (Kaggle CelebA-HQ).")
            return True
    except Exception as exc:
        print(f"[2A] Method 1 failed: {exc}")

    try:
        if method_2a_lfw():
            print("[2A] Success via Method 3 (LFW fallback).")
            return True
    except Exception as exc:
        print(f"[2A] Method 3 failed: {exc}")

    print("[2A] Failed to reach target. Manual fallback URLs:")
    print("https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq")
    print("Target path: D:/deepfake_detection/data/real")
    return False


def method_2b_kaggle_fake() -> bool:
    print("[2B] Trying Method 1: 140k Real and Fake Faces via Kaggle")
    if not has_kaggle_token():
        print("[2B] Kaggle token missing.")
        print("URL: https://www.kaggle.com/settings/account")
        print("Save token to: C:/Users/<your-user>/.kaggle/kaggle.json")
        input("Press Enter after placing kaggle.json, or Ctrl+C to stop...")
        return False

    ensure_dir(FAKE_DIR)
    ensure_dir(TMP_FAKE)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            "xhlulu/140k-real-and-fake-faces",
            "-p",
            str(TMP_FAKE),
            "--unzip",
        ],
        check=True,
    )

    copied = 0
    for file_path in glob.glob(str(TMP_FAKE / "**/fake/**/*.jpg"), recursive=True):
        src = Path(file_path)
        dst = FAKE_DIR / src.name
        if not dst.exists():
            shutil.copy(src, dst)
            copied += 1

    print(f"[2B] Method 1 succeeded. Copied {copied} images. fake/ now has {count_images(FAKE_DIR)} images")
    return count_images(FAKE_DIR) >= 1000


def step_2b_fake_data() -> bool:
    current = count_images(FAKE_DIR)
    if current >= 1000:
        print(f"[2B] fake/ already has {current} images. Skipped.")
        return True

    try:
        if method_2b_kaggle_fake():
            print("[2B] Success via Method 1 (Kaggle 140k fake split).")
            return True
    except Exception as exc:
        print(f"[2B] Method 1 failed: {exc}")

    print("[2B MANUAL FALLBACK]")
    print("1. Go to: https://dfdc.ai/")
    print("   or: https://www.kaggle.com/datasets/humananalog/deepfake-detection-challenge-sample")
    print("2. Download the preview/sample zip")
    print("3. Extract face crops to: D:/deepfake_detection/data/fake/")
    print("4. Resume with: d:/deepfake_detection/.venv/Scripts/python.exe scripts/preflight_check.py")
    input("Press Enter to continue after manual download...")
    return False


def step_2c_build_atn_splits() -> bool:
    ensure_dir(TRAIN_DIR)
    ensure_dir(VAL_DIR)

    all_images = list(REAL_DIR.glob("*.jpg")) + list(REAL_DIR.glob("*.png"))
    random.seed(42)
    random.shuffle(all_images)

    train_images = all_images[:4000]
    val_images = all_images[4000:5000]

    existing_train = len(list(TRAIN_DIR.iterdir()))
    existing_val = len(list(VAL_DIR.iterdir()))

    if existing_train < 3500:
        copied_train = 0
        for img in train_images:
            dst = TRAIN_DIR / img.name
            if not dst.exists():
                shutil.copy(img, dst)
                copied_train += 1
        print(f"[2C] Copied {copied_train} -> atn_train/")
    else:
        print(f"[2C] atn_train/ already has {existing_train} images - skipped")

    if existing_val < 900:
        copied_val = 0
        for img in val_images:
            dst = VAL_DIR / img.name
            if not dst.exists():
                shutil.copy(img, dst)
                copied_val += 1
        print(f"[2C] Copied {copied_val} -> atn_val/")
    else:
        print(f"[2C] atn_val/ already has {existing_val} images - skipped")

    return count_images(TRAIN_DIR) >= 3500 and count_images(VAL_DIR) >= 900


def main() -> int:
    ok_real = step_2a_real_data()
    ok_fake = step_2b_fake_data()
    ok_atn = step_2c_build_atn_splits()

    if ok_real and ok_fake and ok_atn:
        print("[STEP 2] ✓ Done")
        return 0

    print("[STEP 2] ✗ Failed: One or more data targets not met")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())