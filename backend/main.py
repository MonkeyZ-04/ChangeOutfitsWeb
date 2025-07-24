import torch
import cv2
import numpy as np
import pathlib
import requests
import tqdm
from PIL import Image
from io import BytesIO
import os
import io

from rembg import remove
from segment_anything import sam_model_registry, SamPredictor

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

# --- ค่าคงที่และการโหลดโมเดล ---
DEVICE = "cpu"
CHECKPOINT_NAME = "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
CHECKPOINT_URL = f"https://dl.fbaipublicfiles.com/segment_anything/{CHECKPOINT_NAME}"
CHECKPOINT_PATH = pathlib.Path("models") / CHECKPOINT_NAME

# --- ดาวน์โหลด Checkpoint หากยังไม่มี ---
CHECKPOINT_PATH.parent.mkdir(exist_ok=True)
if not CHECKPOINT_PATH.exists():
    print(f"กำลังดาวน์โหลด SAM checkpoint ไปยัง {CHECKPOINT_PATH}...")
    with requests.get(CHECKPOINT_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(CHECKPOINT_PATH, "wb") as f, tqdm.tqdm(
            total=total, unit="B", unit_scale=True, desc=CHECKPOINT_NAME
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    print("ดาวน์โหลดเสร็จสิ้น")

# --- โหลดโมเดล SAM ---
sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT_PATH))
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

# --- FastAPI App Initialization ---
app = FastAPI()

# --- CORS Middleware เพื่อให้ Frontend เรียกใช้ได้ ---
origins = ["*"]  # อนุญาตทุก origins เพื่อความง่าย (ใน Production ควรจำกัด)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- ฟังก์ชันประมวลผลภาพหลัก (จากไฟล์ try.ipynb) ---
def process_image_from_notebook(image_bytes: bytes) -> bytes:
    try:
        # --- ขั้นตอนที่ 1: โหลดภาพจาก bytes และแปลงเป็น RGB ---
        nparr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("ไม่สามารถถอดรหัสรูปภาพได้")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # --- ขั้นตอนที่ 2: ลบพื้นหลังด้วย rembg ---
        img_pil = Image.fromarray(rgb).convert("RGBA")
        removed_output = remove(img_pil)

        if isinstance(removed_output, bytes):
            removed_pil = Image.open(BytesIO(removed_output)).convert("RGBA")
        else:
            removed_pil = removed_output.convert("RGBA")

        # สำหรับ SAM ต้องใช้ภาพ 3-channel (RGB) โดยรวม background โปร่งใสกับสีขาว
        rgba_nobg_array = np.array(removed_pil)
        alpha_mask = rgba_nobg_array[..., 3:] / 255.0
        rgb_nobg = (rgba_nobg_array[..., :3] * alpha_mask + (1 - alpha_mask) * 255).astype(np.uint8)

        # --- ขั้นตอนที่ 3: ใช้ SAM เพื่อหา Mask ของเสื้อด้วย Bounding Box ---
        h, w, _ = rgb_nobg.shape
        # Bbox คร่าวๆ อาจจะต้องปรับแก้ตามสัดส่วนภาพ
        bbox_shirt = np.array([int(w * 0.1), int(h * 0.4), int(w * 0.9), int(h * 0.95)])

        predictor.set_image(rgb_nobg)
        masks, scores, _ = predictor.predict(
            box=bbox_shirt[None, :],
            multimask_output=True
        )
        
        # เลือก Mask ที่ดีที่สุด (มี score สูงสุด หรือ พื้นที่ใหญ่สุด)
        # ใน try.ipynb เลือกอันที่มีพื้นที่ใหญ่สุด ซึ่งมักจะให้ผลดีกว่า
        mask_areas = [np.sum(m) for m in masks]
        best_mask_idx = np.argmax(mask_areas)
        shirt_mask = masks[best_mask_idx]

        # --- ขั้นตอนที่ 4: ขยาย Mask (Dilate) เพื่อให้ขอบเนียน ---
        # Kernel size และ iterations เหมือนใน try.ipynb (หรือปรับให้เหมาะสม)
        kernel = np.ones((7, 7), np.uint8)
        mask_big = cv2.dilate(shirt_mask.astype(np.uint8), kernel, iterations=2)

        # --- ขั้นตอนที่ 5: สร้างภาพสุดท้าย (RGBA) ที่เสื้อโปร่งใส ---
        # สร้าง Alpha channel ใหม่: ส่วนที่เป็นเสื้อ = 0 (โปร่งใส), ส่วนอื่น = 255 (ทึบ)
        alpha_final = (1 - mask_big) * 255
        # รวมภาพ RGB ที่ไม่มีพื้นหลัง กับ Alpha channel ใหม่
        rgba_final = np.dstack([rgb_nobg, alpha_final.astype(np.uint8)])
        
        # แปลงกลับเป็น BGRA เพื่อให้ cv2.imencode ทำงานได้ถูกต้องกับ PNG
        bgra_final = cv2.cvtColor(rgba_final, cv2.COLOR_RGBA2BGRA)

        # เข้ารหัสเป็น bytes ของไฟล์ PNG
        is_success, buffer = cv2.imencode(".png", bgra_final)
        if not is_success:
            raise ValueError("ไม่สามารถเข้ารหัสภาพเป็น PNG ได้")
            
        return buffer.tobytes()

    except Exception as e:
        print(f"เกิดข้อผิดพลาดใน process_image_from_notebook: {e}")
        raise

# --- API Endpoint ---
@app.post("/process-image")
async def process_image_endpoint(file: UploadFile = File(...)):
    """
    รับรูปภาพ, ทำการตัดเสื้อออกโดยใช้หลักการจาก try.ipynb, 
    และส่งภาพที่มีเฉพาะส่วนหัวและคอกลับไป
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="ไฟล์ที่ส่งมาไม่ใช่รูปภาพ")

    try:
        image_bytes = await file.read()
        processed_image_bytes = process_image_from_notebook(image_bytes)
        
        return StreamingResponse(io.BytesIO(processed_image_bytes), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดระหว่างการประมวลผล: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Outfit Try-On API (Updated)"}