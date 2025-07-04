from __future__ import annotations
import cv2, numpy as np
from functools import lru_cache
from typing import List, Dict

# 3rd‑party
from rembg import remove
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image
from io import BytesIO
from pathlib import Path



# -------------------------------------------------
# 🔧 — UTILITY FUNCTIONS (คงไว้ตาม notebook)
# -------------------------------------------------

@lru_cache(maxsize=1)
def _load_sam_model(device: str = "cpu"):
    """โหลด Segment-Anything พร้อมระบุ path weight ให้ถูกต้อง"""
    root_dir = Path(__file__).resolve().parent       
    ckpt = root_dir / "models" / "sam_vit_h_4b8939.pth"

    if not ckpt.exists():
        raise FileNotFoundError(f"SAM checkpoint not found at {ckpt}")

    sam = sam_model_registry["vit_h"](checkpoint=str(ckpt))
    sam.to(device=device)
    return sam

def compute_relative_bbox(bbox_px: np.ndarray, orig_size: tuple[int, int]) -> np.ndarray:
    """พิกัด absolute → relative (0–1) """
    ow, oh = orig_size
    return bbox_px / np.array([ow, oh, ow, oh], dtype=float)

def compute_absolute_bbox(bbox_rel: np.ndarray, new_size: tuple[int, int]) -> np.ndarray:
    """พิกัด relative → absolute (pixel) """
    nw, nh = new_size
    bbox = (bbox_rel * np.array([nw, nh, nw, nh], dtype=float)).astype(int)
    # clamp ให้อยู่ในกรอบภาพ
    bbox[0] = max(0, bbox[0]);  bbox[1] = max(0, bbox[1])
    bbox[2] = min(nw, bbox[2]); bbox[3] = min(nh, bbox[3])
    return bbox

def crop_by_percent_np(img: np.ndarray,
                       top_pct: float = 0.0, bottom_pct: float = 0.3,
                       left_pct: float = 0.2, right_pct: float = 0.2) -> np.ndarray:
    """ครอปภาพตามสัดส่วน (0–1) จากขอบ"""
    h, w = img.shape[:2]
    top    = int(h * top_pct)
    bottom = int(h * (1 - bottom_pct))
    left   = int(w * left_pct)
    right  = int(w * (1 - right_pct))
    return img[top:bottom, left:right]


# -------------------------------------------------
# 🧠 — HELPERS สำหรับ pipeline
# -------------------------------------------------
@lru_cache(maxsize=1)
def _load_sam_model(device: str = "cpu"):
    """
    โหลด Segment-Anything; รองรับทั้ง vit_b และ vit_h
    เลือก checkpoint ที่มีอยู่จริงในโฟลเดอร์ models/
    """
    root = Path(__file__).resolve().parent        # backend/app
    model_dir = root / "models"

    # mapping: arch -> checkpoint-file
    candidates = {
        "vit_b": model_dir / "sam_vit_b_01ec64.pth",
        "vit_h": model_dir / "sam_vit_h_4b8939.pth",
    }

    for arch, ckpt_path in candidates.items():
        if ckpt_path.exists():
            sam = sam_model_registry[arch](checkpoint=str(ckpt_path))
            sam.to(device=device)
            return sam

    raise FileNotFoundError(
        f"No SAM checkpoint found in {model_dir}. "
        f"Expected one of: {[p.name for p in candidates.values()]}"
    )

def _pick_shirt_mask(masks: List[Dict], size: tuple[int, int]) -> np.ndarray | None:
    """เลือกมาสก์ที่น่าจะเป็น 'เสื้อ' — heuristic: 
       ① มี area ใหญ่สุด และ ② centroid อยู่ครึ่งล่างของภาพ"""
    h, _ = size
    candidates = []
    for m in masks:
        seg = m["segmentation"]
        ys, xs = np.where(seg)
        if len(ys) == 0:
            continue
        area     = seg.sum()
        centroid = ys.mean()
        if centroid > h * 0.4:   # อยู่ครึ่งล่าง
            candidates.append((area, seg))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1].astype(bool)

def _mask_from_bbox(img_rgb: np.ndarray,
                    bbox_px: np.ndarray,
                    sam_model,
                    dilate_kernel: tuple[int,int]=(10,10),
                    dilate_iter: int = 2) -> np.ndarray:
    """
    ใช้ SamPredictor + bbox absolute (x0,y0,x1,y1) สร้าง mask เสื้อ
    คืนค่า bool mask (H×W)
    """
    predictor = SamPredictor(sam_model)
    predictor.set_image(img_rgb)

    # SamPredictor ต้องการ shape (1,4)
    masks, scores, _ = predictor.predict(
        box=bbox_px[None, :],
        point_coords=None,
        point_labels=None,
        multimask_output=False
    )
    mask = masks[0].astype(np.uint8)

    if dilate_iter > 0:
        kernel = np.ones(dilate_kernel, np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    return mask.astype(bool)

# -------------------------------------------------
# 🎯 — MAIN ENTRY
# -------------------------------------------------
def process_image(
        img_bgr: np.ndarray,
        shirt_bbox_px: tuple[int,int,int,int] | None = None,
        shirt_bbox_rel: tuple[float,float,float,float] | None = None
    ) -> np.ndarray:
    """ขั้นตอนเต็ม: BG → SAM mask เสื้อ → ลบเสื้อ → ลบ BG รอบหัว‑คอ → ครอปเล็กน้อย"""

    # 1) OpenCV → PIL RGBA
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # 2) ลบพื้นหลังขั้นแรก (rembg)
    first_pass = remove(img_pil)
    if isinstance(first_pass, bytes):
        first_pass = Image.open(BytesIO(first_pass)).convert("RGBA")

    rgba = np.array(first_pass)
    rgb  = rgba[..., :3]
    alpha= rgba[..., 3:4] / 255.0
    # รวมพื้นหลังโปร่งใสกับสีขาว → rgb_nobg
    rgb_nobg = (rgb * alpha + (1 - alpha) * 255).astype(np.uint8)

    # 3) SAM → หา mask เสื้อ
    sam = _load_sam_model()

    # ► 3.a  แปลง bbox_rel → abs px ถ้ามี
    if shirt_bbox_rel is not None and shirt_bbox_px is None:
        h, w = rgb_nobg.shape[:2]
        shirt_bbox_px = compute_absolute_bbox(np.array(shirt_bbox_rel), (w, h))

    if shirt_bbox_px is not None:
        # --- ใช้ SamPredictor + bbox ---
        shirt_mask = _mask_from_bbox(rgb_nobg,
                                     np.array(shirt_bbox_px, dtype=int),
                                     sam_model=sam,
                                     dilate_kernel=(10,10),
                                     dilate_iter=2)
    else:
        # --- ใช้ Auto-Mask แบบเดิม ---
        mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            min_mask_region_area=512
        )
        auto_masks = mask_generator.generate(rgb_nobg)
        shirt_mask = _pick_shirt_mask(auto_masks, rgb_nobg.shape[:2])

    if shirt_mask is None:
        # หาเสื้อไม่เจอ → คืนภาพ BG-removed
        return cv2.cvtColor(np.array(first_pass), cv2.COLOR_RGBA2BGRA)

    # 4) ขยาย mask & ทำ alpha เสื้อโปร่ง
    kernel      = np.ones((10, 10), np.uint8)
    mask_big    = cv2.dilate(shirt_mask.astype(np.uint8), kernel, iterations=2)
    alpha_full  = (1 - mask_big) * 255
    rgba_clean  = np.dstack([rgb_nobg, alpha_full.astype(np.uint8)])

    # 5) ลบ BG รอบหัว‑คอ (rembg อีกครั้ง)
    clean_pil   = Image.fromarray(rgba_clean, mode="RGBA")
    headneck    = remove(clean_pil)
    if isinstance(headneck, bytes):
        headneck = Image.open(BytesIO(headneck)).convert("RGBA")

    final_rgba  = np.array(headneck)

    # 6) ครอปเล็กน้อยเพื่อตัดขอบด้านล่าง + ซ้าย/ขวา
    final_rgba  = crop_by_percent_np(final_rgba,
                                     top_pct=0.0, bottom_pct=0.3,
                                     left_pct=0.2, right_pct=0.2)

    # 7) PIL RGBA → OpenCV BGRA
    return cv2.cvtColor(final_rgba, cv2.COLOR_RGBA2BGRA)

