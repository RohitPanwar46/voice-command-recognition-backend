import json
import os
import tempfile
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf

MODEL_DIR = Path(__file__).parent / "model"

SAMPLE_RATE = 16000
TARGET_SAMPLES = 16000
N_MFCC = 13

# ============================================================
# Load model + label mapping + normalization stats ONCE at startup
# ============================================================

model = tf.keras.models.load_model(MODEL_DIR / "cnn_voice_command_model.keras")

with open(MODEL_DIR / "classes.json", "r") as f:
    class_to_index = json.load(f)
id_to_class = {v: k for k, v in class_to_index.items()}

with open(MODEL_DIR / "feature_stats.json", "r") as f:
    stats = json.load(f)
FEATURE_MEAN = stats["mean"]
FEATURE_STD = stats["std"]


# ============================================================
# Preprocessing - must mirror prepare_dataset.py exactly
# ============================================================

def normalize_peak(samples):
    max_amplitude = np.max(np.abs(samples))
    if max_amplitude > 0:
        samples = samples / max_amplitude
    return samples


def fit_to_target_length(samples):
    if len(samples) == TARGET_SAMPLES:
        return samples
    if len(samples) < TARGET_SAMPLES:
        pad_total = TARGET_SAMPLES - len(samples)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(samples, (pad_left, pad_right), mode="constant")
    peak_idx = int(np.argmax(np.abs(samples)))
    start = peak_idx - TARGET_SAMPLES // 2
    start = max(0, min(start, len(samples) - TARGET_SAMPLES))
    return samples[start:start + TARGET_SAMPLES]


def extract_mfcc(audio):
    return librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC).astype(np.float32)


def predict_from_audio_bytes(audio_bytes, filename_hint="audio.webm"):
    # librosa's ffmpeg fallback (needed for webm/ogg/mp3) only kicks in when
    # given a real file PATH - it shells out to the ffmpeg binary, which
    # needs an actual file on disk, not an in-memory BytesIO object. So we
    # write the upload to a temp file first, then load from that path.
    suffix = Path(filename_hint).suffix or ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        audio, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    finally:
        os.remove(tmp_path)

    audio = normalize_peak(audio)
    audio = fit_to_target_length(audio)

    mfcc = extract_mfcc(audio)
    mfcc = (mfcc - FEATURE_MEAN) / FEATURE_STD
    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    probs = model(mfcc, training=False).numpy()[0]
    pred_id = int(np.argmax(probs))
    confidence = float(probs[pred_id])

    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [{"label": id_to_class[int(i)], "confidence": float(probs[i])} for i in top3_idx]

    return {
        "label": id_to_class[pred_id],
        "confidence": confidence,
        "top3": top3,
    }