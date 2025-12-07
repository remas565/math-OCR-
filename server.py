from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pix2tex.cli import LatexOCR
from PIL import Image
import io
import uvicorn
import os
import time
from datetime import datetime

# ---------------------------------
# إنشاء تطبيق FastAPI
# ---------------------------------
app = FastAPI()

# ---------------------------------
# إعداد CORS علشان الواجهة تقدر تتصل من أي دومين
# ---------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # لو حابة تقفليه لاحقًا نضبطه
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------
# فولدر اللوقز
# ---------------------------------
os.makedirs("logs", exist_ok=True)

# ---------------------------------
# تحميل الموديل مرة واحدة فقط
# pix2tex راح يتكفل بتنزيل الـ weights داخلياً
# ---------------------------------
print("🔁 Initializing LatexOCR model...")
model = LatexOCR()
print("✅ Model loaded successfully.")

# ---------------------------------
# روت بسيط للتأكد إن السيرفر شغال
# ---------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "LatexOCR backend is running 🎉"}

# ---------------------------------
# endpoint: /predict
# يستقبل صورة ويرجع LaTeX + أزمنة التنفيذ
# ---------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ⏱ وقت رفع الملف
    upload_start = time.time()
    contents = await file.read()
    upload_end = time.time()
    upload_time_ms = int((upload_end - upload_start) * 1000)

    # تحويل المحتوى لصورة
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # حفظ نسخة من الصورة في logs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    saved_image_path = f"logs/{timestamp}_image.png"
    image.save(saved_image_path)

    # ⏱ وقت المعالجة بالموديل
    processing_start = time.time()
    try:
        latex_result = model(image)
    except Exception as e:
        latex_result = f"ERROR: {str(e)}"
    processing_end = time.time()

    processing_time_ms = int((processing_end - processing_start) * 1000)
    total_time_ms = upload_time_ms + processing_time_ms

    # حفظ لوق في ملف txt
    log_path = f"logs/{timestamp}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Filename: {file.filename}\n")
        f.write(f"Upload Time: {upload_time_ms} ms\n")
        f.write(f"Processing Time: {processing_time_ms} ms\n")
        f.write(f"Total Time: {total_time_ms} ms\n")
        f.write(f"Output LaTeX: {latex_result}\n")
        f.write(f"Saved Image: {saved_image_path}\n")

    # نتيجة الـ API
    return {
        "latex": latex_result,
        "upload_time": upload_time_ms,
        "processing_time": processing_time_ms,
        "total_time": total_time_ms,
        "log_file": log_path,
    }


# ---------------------------------
# تشغيل محلي (Railway يستخدم CMD الخاص فيه غالبًا)
# ---------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
