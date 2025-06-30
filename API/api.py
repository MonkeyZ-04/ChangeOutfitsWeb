# api.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from processor import process_image
import io 

app = FastAPI(
    title="Outfit-Remover API",
    description="รับรูป → คืนรูปที่ตัดเสื้อและพื้นหลังออกแล้ว"
)

# อนุญาตทุกต้นทาง (สำหรับ Dev เท่านั้น)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
@app.post("/remove-outfit")
async def remove_outfit(file: UploadFile = File(...)):
    # ตรวจสอบนามสกุล
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="ต้องอัปโหลดไฟล์รูปภาพเท่านั้น")
    content = await file.read()
    try:
        result_bytes = process_image(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {e}")
    return StreamingResponse(io.BytesIO(result_bytes), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
