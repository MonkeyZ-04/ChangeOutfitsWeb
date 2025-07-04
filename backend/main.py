from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from io import BytesIO

from .test import process_image

app = FastAPI(title="Replace Clothes API")

# ► อนุญาตให้ frontend (origin ใดก็ได้หรือระบุโดเมน) เรียก API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/process-image", summary="รับรูปแล้วคืนรูปหลังประมวลผล")
async def process_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(status_code=400, detail="Only PNG/JPEG is supported")

    # ▸ อ่านเป็น numpy array
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    out_bgra = process_image(img_bgr, shirt_bbox_rel=(0,0.6,1,1)) 

    # ▸ เข้ารหัสกลับเป็น PNG bytes
    success, buf = cv2.imencode(".png", out_bgra)
    if not success:
        raise HTTPException(status_code=500, detail="Encoding failed")

    return StreamingResponse(BytesIO(buf.tobytes()),
                             media_type="image/png")

