from pathlib import Path
import random
import shutil

base = Path("D:/deepfake_detection")
real = base / "data/real"
train = base / "data/atn_train"
val = base / "data/atn_val"
real.mkdir(parents=True, exist_ok=True)
train.mkdir(parents=True, exist_ok=True)
val.mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

lfw = fetch_lfw_people(min_faces_per_person=1, resize=1.0, data_home="D:/deepfake_detection/data/tmp_lfw", color=True, download_if_missing=True)
saved = 0
for i, img in enumerate(lfw.images):
    out = real / f"lfw_{i:05d}.jpg"
    if not out.exists():
        img_uint8 = (img * 255).astype(np.uint8)
        cv2.imwrite(str(out), cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
        saved += 1

all_images = list(real.glob("*.jpg")) + list(real.glob("*.png"))
random.seed(42)
random.shuffle(all_images)
train_images = all_images[:4000]
val_images = all_images[4000:5000]

copied_train = 0
for img in train_images:
    dst = train / img.name
    if not dst.exists():
        shutil.copy(img, dst)
        copied_train += 1

copied_val = 0
for img in val_images:
    dst = val / img.name
    if not dst.exists():
        shutil.copy(img, dst)
        copied_val += 1

print(f"LFW_SAVED={saved}")
print(f"ATN_TRAIN_COPIED={copied_train}")
print(f"ATN_VAL_COPIED={copied_val}")
print(f"REAL_TOTAL={len([f for f in real.rglob('*') if f.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}])}")
print(f"TRAIN_TOTAL={len([f for f in train.rglob('*') if f.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}])}")
print(f"VAL_TOTAL={len([f for f in val.rglob('*') if f.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}])}")
