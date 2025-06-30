import torch
import cv2
import numpy as np
import pathlib
import requests
import tqdm
from PIL import Image
from io import BytesIO
from rembg import remove
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# ---------------------------------------------
#  Configuration
# ---------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# vit B
# CHECKPOINT_NAME = "sam_vit_b_01ec64.pth"

#vit H
CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"
CHECKPOINT_URL  = f"https://dl.fbaipublicfiles.com/segment_anything/{CHECKPOINT_NAME}"

# Directories
INPUT_DIR  = pathlib.Path("OUTFITS")
OUTPUT_DIR = pathlib.Path("RESULT_CUT_BG")
MODEL_DIR  = pathlib.Path("models")

# BBox for shirt prompt (ปรับตามภาพถ้าจำเป็น)
bbox_shirt = np.array([60, 240, 630, 650])

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------
#  Download SAM checkpoint if not exists
# ---------------------------------------------
weights = MODEL_DIR / CHECKPOINT_NAME
if not weights.exists():
    with requests.get(CHECKPOINT_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(weights, "wb") as f, tqdm.tqdm(total=total, unit="B", unit_scale=True,
                                                 desc=CHECKPOINT_NAME) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

# ---------------------------------------------
#  Load SAM model and mask generator
# ---------------------------------------------
sam = sam_model_registry["vit_h"](checkpoint=str(weights))
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side      = 32,
    min_mask_region_area = 512,
)
predictor = SamPredictor(sam)

# ---------------------------------------------
#  Processing function
# ---------------------------------------------
def process_image(img_path: pathlib.Path):
    # 1) Load image (BGR -> RGB)
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"❌ Cannot read {img_path}")
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 2) Pre-remove background (rembg)
    img_pil = Image.fromarray(rgb).convert("RGBA")
    removed = remove(img_pil)
    if isinstance(removed, bytes):
        removed = Image.open(BytesIO(removed)).convert("RGBA")
    rgba = np.array(removed)
    alpha_mask = rgba[..., 3:] / 255.0
    rgb_nobg = (rgba[..., :3] * alpha_mask + (1 - alpha_mask) * 255).astype(np.uint8)

    # 3) SAM-based shirt mask
    predictor.set_image(rgb_nobg)
    masks, scores, _ = predictor.predict(
        box=bbox_shirt[None, :],
        multimask_output=True
    )
    # เลือก mask ที่มีพื้นที่ใหญ่สุด
    areas = [m.sum() for m in masks]
    shirt_mask = masks[int(np.argmax(areas))]

    # 4) ขยาย mask เล็กน้อย
    kernel = np.ones((7, 7), np.uint8)
    mask_big = cv2.dilate(shirt_mask.astype(np.uint8), kernel, iterations=2)

    # 5) สร้างภาพ RGBA สำหรับ head&neck (เสื้อ = โปร่งใส)
    alpha_full = (1 - mask_big) * 255
    rgba_clean = np.dstack([rgb_nobg, alpha_full.astype(np.uint8)])

    # 6) สุดท้าย remove background อีกครั้ง เพื่อเก็บเฉพาะ head&neck
    tmp_pil = Image.fromarray(rgba_clean).convert("RGBA")
    final = remove(tmp_pil)
    if isinstance(final, bytes):
        final = Image.open(BytesIO(final)).convert("RGBA")

    # 7) Save result
    out_path = OUTPUT_DIR / f"{img_path.stem}_headneck.png"
    final.save(out_path)
    print(f"✔️ Saved → {out_path}")

# ---------------------------------------------
#  Loop through all images in OUTFITS
# ---------------------------------------------
if __name__ == "__main__":
    for img_file in INPUT_DIR.iterdir():
        if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            process_image(img_file)
