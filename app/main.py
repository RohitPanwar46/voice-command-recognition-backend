from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.inference import predict_from_audio_bytes

app = FastAPI(title="Voice Command Recognition API")

# Tighten allow_origins to your actual Vercel domain once deployed,
# e.g. ["https://your-frontend.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://voice-command-recognition-system.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    result = predict_from_audio_bytes(audio_bytes, filename_hint=file.filename or "audio.webm")
    return result