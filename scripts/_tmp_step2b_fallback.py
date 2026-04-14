from pathlib import Path
import random
import cv2
import numpy as np

base = Path("D:/deepfake_detection")
real_dir = base / "data/real"
fake_dir = base / "data/fake"
fake_dir.mkdir(parents=True, exist_ok=True)

real_images = [p for p in real_dir.glob("*.jpg")] + [p for p in real_dir.glob("*.png")]
random.seed(42)
random.shuffle(real_images)

needed = max(0, 1000 - len([f for f in fake_dir.rglob('*') if f.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}]))
created = 0

for i, path in enumerate(real_images):
    if created >= needed:
        break
    img = cv2.imread(str(path))
    if img is None:
        continue

    # Heavy artifacts to create stand-in synthetic negatives for pipeline smoke/testing
    h, w = img.shape[:2]
    img2 = cv2.GaussianBlur(img, (9, 9), sigmaX=2.5)
    noise = np.random.normal(0, 18, img2.shape).astype(np.int16)
    img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img2 = cv2.resize(img2, (max(32, w // 2), max(32, h // 2)), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_LINEAR)
    encode_ok, enc = cv2.imencode('.jpg', img2, [int(cv2.IMWRITE_JPEG_QUALITY), 35])
    if not encode_ok:
        continue
    out = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    out_path = fake_dir / f"augfake_{i:05d}.jpg"
    cv2.imwrite(str(out_path), out)
    created += 1

count_fake = len([f for f in fake_dir.rglob('*') if f.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}])
print("WARNING: Using augmented-real as stand-in fakes. Replace with actual deepfakes for production training.")
print(f"FAKE_CREATED={created}")
print(f"FAKE_TOTAL={count_fake}")
