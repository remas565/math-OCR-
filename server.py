from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pix2tex.cli import LatexOCR
from PIL import Image
import io, os, time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("logs", exist_ok=True)

# -------------------------
# Load Lightweight Model
# -------------------------
print("🔁 Loading lightweight Pix2TeX model...")

model = LatexOCR(
    checkpoint="https://github.com/Mathpix/pix2tex/releases/download/v0.1.0/weights-mini.pth"
)

print("✔ Model loaded with lightweight weights!")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()
    img_bytes = await file.read()

    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    try:
        result = model(image)
    except Exception as e:
        result = f"ERROR: {e}"

    return {
        "latex": result,
        "processing_time": int((time.time() - start) * 1000)
    }


if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
