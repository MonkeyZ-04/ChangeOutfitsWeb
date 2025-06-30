# processor.py
import io
import torch, cv2, numpy as np
from PIL import Image
from rembg import remove
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"
CHECKPOINT_URL  = f"https://dl.fbaipublicfiles.com/segment_anything/{CHECKPOINT_NAME}"

# โหลดโมเดลครั้งเดียวตอนเริ่ม
# (คุณอาจย้ายส่วนดาวน์โหลด checkpoint ออกไปก่อนหรือใส่ไว้ที่นี่ก็ได้)
sam = sam_model_registry["vit_h"](checkpoint=str("models/"+CHECKPOINT_NAME))
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(
    sam, points_per_side=32, min_mask_region_area=512
)
predictor = SamPredictor(sam)

def process_image(input_bytes: bytes) -> bytes:
    # 1) อ่านภาพจาก bytes → RGB numpy
    img_pil = Image.open(io.BytesIO(input_bytes)).convert("RGB")
    rgb = np.array(img_pil)

    # 2) ลบ BG ด้วย rembg → ได้ RGBA
    rgba = remove(img_pil)
    if isinstance(rgba, bytes):
        rgba = Image.open(io.BytesIO(rgba)).convert("RGBA")
    rgba = np.array(rgba)
    alpha_mask = rgba[..., 3:] / 255.0
    rgb_nobg = (rgba[..., :3] * alpha_mask + (1 - alpha_mask) * 255).astype(np.uint8)

    # 3) สร้าง mask อัตโนมัติ (ถ้าต้องการ) หรือใช้ box prompt
    #    สมมติเลือก box เดิมในตัวอย่าง
    bbox_shirt = np.array([60, 240, 630, 650])  # ปรับตามภาพจริงหรือรับจาก client
    predictor.set_image(rgb_nobg)
    masks, scores, _ = predictor.predict(
        box=bbox_shirt[None, :],
        multimask_output=True
    )
    # เอา mask ที่ใหญ่สุด
    shirt_mask = masks[np.argmax(masks.reshape(masks.shape[0], -1).sum(axis=1))]

    # 4) ขยาย mask เล็กน้อย
    kernel = np.ones((7,7), np.uint8)
    mask_big = cv2.dilate(shirt_mask.astype(np.uint8), kernel, iterations=2)

    # 5) สร้างภาพ RGBA ที่เสื้อโปร่งใส
    alpha_full = (1 - mask_big) * 255
    rgba_clean = np.dstack([rgb_nobg, alpha_full.astype(np.uint8)])

    # 6) ลบ BG รอบหัว/คอ อีกครั้ง
    headneck_img = Image.fromarray(rgba_clean).convert("RGBA")
    bg_removed = remove(headneck_img)
    if isinstance(bg_removed, bytes):
        bg_removed = Image.open(io.BytesIO(bg_removed)).convert("RGBA")

    # 7) แปลงกลับเป็น bytes (PNG)
    buf = io.BytesIO()
    bg_removed.save(buf, format="PNG")
    return buf.getvalue()
