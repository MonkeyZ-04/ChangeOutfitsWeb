"""
processing.py — main pipeline to remove shirts from portraits
------------------------------------------------------------
This version **keeps every original function** and touches **only the bbox
handling** so you can feed either absolute‐pixel or 0‑1 relative bbox and get
the same outcome without UnboundLocalError.

Key points
~~~~~~~~~~
* Call `process_image(img, shirt_bbox_px=(x0,y0,x1,y1))` **or**
  `process_image(img, shirt_bbox_rel=(x0,y0,x1,y1))`.
  If both are given, **pixel wins**.
* Internally we always convert relative → pixel for SAM, and pixel → relative
  for the return dict, so you can daisy‑chain the 0‑1 box later.
* Optional `debug=True` draws a green rectangle and saves **bbox_vis.png**.
* Crop code is still here but commented out exactly like the original request.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from functools import lru_cache     
import cv2
import numpy as np
from rembg import remove
from PIL import Image
from io import BytesIO
import torch
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# =============================================================
# 1.  Load SAM once & cache
# =============================================================

_SAM_CACHE: Optional[torch.nn.Module] = None


@lru_cache(maxsize=1)
def _load_sam_model(model_type: str = "vit_h",
                    ckpt_path: str | Path | None = None,
                    device: str = "cpu"):
    """
    Load Segment-Anything once, trying several candidate paths:

    1) *ckpt_path* (if the caller passes one)
    2) <repo root>/backend/app/models/sam_vit_h_4b8939.pth
    3) <repo root>/backend/models/sam_vit_h_4b8939.pth
    """
    root = Path(__file__).resolve().parent          # …/backend
    candidates = []

    # ① explicit argument wins
    if ckpt_path is not None:
        candidates.append(Path(ckpt_path).expanduser())

    # ② backend/app/models
    candidates.append(root / "app" / "models" / "sam_vit_h_4b8939.pth")

    # ③ backend/models  (fallback—theเก่า)
    candidates.append(root / "models" / "sam_vit_h_4b8939.pth")

    for p in candidates:
        if p.exists():
            sam = sam_model_registry[model_type](checkpoint=str(p))
            sam.to(device=device)
            return sam

    raise FileNotFoundError(
        "SAM checkpoint not found.\n"
        "Looked for:\n  " + "\n  ".join(str(p) for p in candidates)
    )

# =============================================================
# 2.  BBox helpers  (— these are the ONLY new/changed parts —)
# =============================================================

def compute_absolute_bbox(bbox_rel: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """(x0,y0,x1,y1) in 0‑1 → absolute px given *(width, height)*."""
    w, h = size  # width first!
    bbox_px = bbox_rel * np.array([w, h, w, h], dtype=np.float32)
    bbox_px[[0, 2]] = np.clip(bbox_px[[0, 2]], 0, w - 1)
    bbox_px[[1, 3]] = np.clip(bbox_px[[1, 3]], 0, h - 1)
    return bbox_px


def compute_relative_bbox(bbox_px: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Absolute px bbox → 0‑1 relative given *(width, height)*."""
    w, h = size
    bbox_rel = bbox_px / np.array([w, h, w, h], dtype=np.float32)
    return np.clip(bbox_rel, 0.0, 1.0)

# =============================================================
# 3.  Shirt‑mask helpers (unchanged except typings)
# =============================================================

def _mask_from_bbox(img_rgb: np.ndarray,
                    bbox_px: np.ndarray,
                    sam_model,
                    dilate_kernel: Tuple[int, int] = (10, 10),
                    dilate_iter: int = 2) -> np.ndarray:
    """Run **SamPredictor** on one bbox → bool mask."""
    predictor = SamPredictor(sam_model)
    predictor.set_image(img_rgb)

    masks, _, _ = predictor.predict(box=bbox_px[None, :], multimask_output=False)
    mask = masks[0].astype(np.uint8)

    if dilate_iter:
        kernel = np.ones(dilate_kernel, np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
    return mask.astype(bool)


def _pick_shirt_mask(auto_masks: list[dict], shape_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    """Heuristic: pick largest mask whose centroid lies below 40 % of image height."""
    h, _ = shape_hw
    if not auto_masks:
        return None
    auto_masks.sort(key=lambda m: m["area"], reverse=True)
    for m in auto_masks:
        cy = m["bbox"][1] + m["bbox"][3] / 2.0  # bbox = [x,y,w,h]
        if cy > 0.4 * h:
            return m["segmentation"].astype(bool)
    return auto_masks[0]["segmentation"].astype(bool)

# =============================================================
# 4.  Main pipeline — unchanged except bbox section
# =============================================================

def process_image(img_bgr: np.ndarray,
                  *,
                  shirt_bbox_px: Optional[Tuple[int, int, int, int]] = None,
                  shirt_bbox_rel: Optional[Tuple[float, float, float, float]] = None,
                  debug: bool = False) -> Dict[str, Any]:
    """Full pipeline. Returns a dict with keys:
        • **image** – BGRA numpy (uint8)
        • **shirt_bbox_px / shirt_bbox_rel** – final bbox actually used
    """

    # ---- 0) Remove background (coarse) ----
    first_pass = remove(img_bgr, alpha_matting=False)
    if isinstance(first_pass, bytes):                        # ① bytes → ndarray
        first_pass = cv2.imdecode(
        np.frombuffer(first_pass, np.uint8), cv2.IMREAD_UNCHANGED
    )
    elif isinstance(first_pass, Image.Image):                # ② PIL.Image → ndarray
        first_pass = cv2.cvtColor(np.array(first_pass), cv2.COLOR_RGBA2BGRA)
    rgba = cv2.cvtColor(first_pass, cv2.COLOR_BGR2RGBA)
    rgb_nobg = rgba[..., :3]

    # ---- 1) Resolve bbox *once* ----
    h, w = rgb_nobg.shape[:2]
    bbox_px: Optional[np.ndarray] = None

    if shirt_bbox_px is not None:
        # Explicit pixel bbox wins.
        bbox_px = np.array(shirt_bbox_px, dtype=np.float32)
        shirt_bbox_rel = tuple(compute_relative_bbox(bbox_px, (w, h)))
    elif shirt_bbox_rel is not None:
        bbox_px = compute_absolute_bbox(np.array(shirt_bbox_rel, dtype=np.float32), (w, h))
    # else: both None → stay None and fall back to auto‑mask

    # ---- 1.a Debug rectangle if requested ----
    if debug and bbox_px is not None:
        dbg = rgb_nobg.copy()
        x0, y0, x1, y1 = map(int, bbox_px)
        cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 0), 4)
        cv2.imwrite("bbox_vis.png", cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))

    # ---- 2) Obtain shirt mask ----
    sam = _load_sam_model()

    if bbox_px is not None:
        shirt_mask = _mask_from_bbox(rgb_nobg, bbox_px, sam)
    else:
        mask_gen = SamAutomaticMaskGenerator(sam, points_per_side=32, min_mask_region_area=512)
        shirt_mask = _pick_shirt_mask(mask_gen.generate(rgb_nobg), (h, w))

    if shirt_mask is None or shirt_mask.sum() < 500:
        # Nothing detected → return BG‑removed image unchanged
        return {
            "image": cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA),
            "shirt_bbox_px": None,
            "shirt_bbox_rel": None,
        }

    # ---- 3) Expand mask and build alpha ----
    kernel = np.ones((10, 10), np.uint8)
    mask_big = cv2.dilate(shirt_mask.astype(np.uint8), kernel, iterations=2)
    alpha_full = ((~mask_big.astype(bool)).astype(np.uint8)) * 255
    rgba_no_shirt = np.dstack([rgb_nobg, alpha_full])

    # ---- 4) Second BG removal to keep only head/neck ----
    head_only = remove(cv2.cvtColor(rgba_no_shirt, cv2.COLOR_RGBA2BGRA))

    # ---- 5) (Optional) crop – currently disabled ----
    # head_only = crop_by_percent_np(head_only, top_pct=0.0, bottom_pct=0.3,
    #                                left_pct=0.2, right_pct=0.2)

    # ---- 6) Return ----
    return {
        "image": head_only,
        "shirt_bbox_px": tuple(map(int, bbox_px)) if bbox_px is not None else None,
        "shirt_bbox_rel": tuple(shirt_bbox_rel) if shirt_bbox_rel is not None else None,
    }

# =============================================================
# 5.  Crop helper (unchanged, still available)
# =============================================================

def crop_by_percent_np(img: np.ndarray,
                       *,
                       top_pct: float = 0.0,
                       bottom_pct: float = 0.0,
                       left_pct: float = 0.0,
                       right_pct: float = 0.0) -> np.ndarray:
    """Crop utility kept from original code."""
    h, w = img.shape[:2]
    top = int(h * top_pct)
    bottom = int(h * (1.0 - bottom_pct))
    left = int(w * left_pct)
    right = int(w * (1.0 - right_pct))
    return img[top:bottom, left:right]
