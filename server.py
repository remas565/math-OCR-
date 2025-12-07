from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pix2tex.cli import LatexOCR
from PIL import Image
import io, uvicorn, os, time, requests
from datetime import datetime

app = FastAPI()

# ----------------------------  CORS  ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------------------------------------------

# Create logs folder if not exists
os.makedirs("logs", exist_ok=True)

# ---------------------- WEIGHTS DOWNLOAD -----------------------
WEIGHTS_PATH = "/tmp/weights.pth"   # <-- مهم جداً ل Railway
DIRECT_URL = "https://drive.google.com/uc?export=download&id=1ZoWviITdtUAbfLs7okEIHPJIMgewnUqs"

def download_weights():
    if os.path.exists(WEIGHTS_PATH):
        print("✔️ Weights already exist in /tmp")
        return

    print("⬇️ Downloading weights...")
    r = requests.get(DIRECT_URL, allow_redirects=True)
    with open(WEIGHTS_PATH, "wb") as f:
        f.write(r.content)

    print("✔️ Weights downloaded successfully!")

download_weights()
# ---------------------------------------------------------------

# Load model once using TEMP path
model = LatexOCR(weights=WEIGHTS_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    upload_start = time.time()
    contents = await file.read()
    upload_end = time.time()
    upload_time_ms = int((upload_end - upload_start) * 1000)

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    saved_image_path = f"logs/{timestamp}_image.png"
    image.save(saved_image_path)

    processing_start = time.time()
    try:
        latex_result = model(image)
    except Exception as e:
        latex_result = f"ERROR: {str(e)}"
    processing_end = time.time()

    processing_time_ms = int((processing_end - processing_start) * 1000)
    total_time_ms = upload_time_ms + processing_time_ms

    log_path = f"logs/{timestamp}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Filename: {file.filename}\n")
        f.write(f"Upload Time: {upload_time_ms} ms\n")
        f.write(f"Processing Time: {processing_time_ms} ms\n")
        f.write(f"Total Time: {total_time_ms} ms\n")
        f.write(f"Output LaTeX: {latex_result}\n")
        f.write(f"Saved Image: {saved_image_path}\n")

    return {
        "latex": latex_result,
        "upload_time": upload_time_ms,
        "processing_time": processing_time_ms,
        "total_time": total_time_ms,
        "log_file": log_path
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
