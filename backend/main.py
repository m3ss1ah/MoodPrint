"""
backend/main.py — FastAPI server for Music Mood LDA
====================================================
Endpoints:
  GET  /api/health            → server + model status
  POST /api/predict           → upload MP3, returns mood + LDA coords + probs
  GET  /api/dataset           → full dataset projection for scatter plot
  GET  /api/discriminants     → LDA weight bars per axis
  GET  /api/metadata          → model metadata (accuracy, variance explained, etc.)
"""

import io
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import librosa
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
MODEL_DIR  = ROOT / "models"
FEATURE_DIM = 180

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Music Mood LDA API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model state ────────────────────────────────────────────────────────────
MODEL = {
    "lda":      None,
    "scaler":   None,
    "le":       None,
    "metadata": None,
    "dataset":  None,
    "loaded":   False,
}

MOOD_COLORS = {
    "Happy":     "#FFD700",
    "Energetic": "#FF4500",
    "Calm":      "#00CED1",
    "Sad":       "#6A5ACD",
}

MOOD_DESCRIPTIONS = {
    "Happy":     "High valence, high arousal. Bright, uplifting, energetic positivity.",
    "Energetic": "Low valence, high arousal. Intense, tense, driving forward motion.",
    "Calm":      "High valence, low arousal. Peaceful, soothing, gentle contentment.",
    "Sad":       "Low valence, low arousal. Melancholic, introspective, subdued.",
}


def load_models():
    """Load all model artefacts from disk."""
    if not MODEL_DIR.exists():
        print(f"⚠  Model directory not found: {MODEL_DIR}")
        return False

    required = ["lda.pkl", "scaler.pkl", "label_encoder.pkl",
                "metadata.json", "dataset_projection.json"]
    missing = [f for f in required if not (MODEL_DIR / f).exists()]
    if missing:
        print(f"⚠  Missing model files: {missing}")
        print("   Run: python train.py")
        return False

    try:
        MODEL["lda"]    = joblib.load(MODEL_DIR / "lda.pkl")
        MODEL["scaler"] = joblib.load(MODEL_DIR / "scaler.pkl")
        MODEL["le"]     = joblib.load(MODEL_DIR / "label_encoder.pkl")

        with open(MODEL_DIR / "metadata.json") as f:
            MODEL["metadata"] = json.load(f)

        with open(MODEL_DIR / "dataset_projection.json") as f:
            MODEL["dataset"] = json.load(f)

        MODEL["loaded"] = True
        print(f"✅ Models loaded from {MODEL_DIR}")
        meta = MODEL["metadata"]
        print(f"   CV accuracy: {meta['cv_accuracy_mean']:.3f} ± {meta['cv_accuracy_std']:.3f}")
        print(f"   Dataset size: {meta['n_samples']} songs")
        return True

    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    load_models()


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (must match train.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    features = []

    # MFCCs (80)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Chroma (24)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # Spectral Contrast (14)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    # Tonnetz (12)
    y_harm = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
    features.extend(np.mean(tonnetz, axis=1))
    features.extend(np.std(tonnetz, axis=1))

    # Mel Spectrogram compressed (24)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    n_bands = 12
    band_size = mel_db.shape[0] // n_bands
    mel_bands = np.array([
        mel_db[i * band_size:(i + 1) * band_size].mean(axis=0)
        for i in range(n_bands)
    ])
    features.extend(np.mean(mel_bands, axis=1))
    features.extend(np.std(mel_bands, axis=1))

    # Spectral shape (10)
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr       = librosa.feature.zero_crossing_rate(y)[0]
    rms       = librosa.feature.rms(y=y)[0]
    for feat in [centroid, bandwidth, rolloff, zcr, rms]:
        features.append(np.mean(feat))
        features.append(np.std(feat))

    # Rhythm (4)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_rate = len(librosa.onset.onset_detect(y=y, sr=sr)) / (len(y) / sr)
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    features.extend([float(tempo), float(np.mean(onset_env)),
                     float(onset_rate), float(np.mean(pulse))])

    # Dynamics (4)
    rms_arr     = librosa.feature.rms(y=y)[0]
    rms_var     = float(np.var(rms_arr))
    dynamic_rng = float(np.max(rms_arr) - np.min(rms_arr))
    silence_ratio = float(np.mean(rms_arr < 1e-4))
    crest_factor  = float(np.max(np.abs(y)) / (np.mean(rms_arr) + 1e-9))
    features.extend([rms_var, dynamic_rng, silence_ratio, crest_factor])

    # Harmonic/Percussive (4)
    y_h, y_p = librosa.effects.hpss(y)
    h_rms = librosa.feature.rms(y=y_h)[0]
    p_rms = librosa.feature.rms(y=y_p)[0]
    features.extend([float(np.mean(h_rms)), float(np.std(h_rms)),
                     float(np.mean(p_rms)), float(np.std(p_rms))])

    # Pitch (4)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[magnitudes > np.percentile(magnitudes, 75)]
    pitch_vals = pitch_vals[pitch_vals > 0]
    if len(pitch_vals) > 0:
        features.extend([float(np.mean(pitch_vals)), float(np.std(pitch_vals)),
                         float(np.median(pitch_vals)), float(np.ptp(pitch_vals))])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])

    arr = np.array(features, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    mood:         str
    confidence:   float
    probabilities: Dict[str, float]
    lda_coords:   Dict[str, float]
    features:     Dict[str, float]
    color:        str
    description:  str
    processing_ms: float


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    n_samples:    Optional[int]
    cv_accuracy:  Optional[float]


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health():
    if not MODEL["loaded"]:
        # Try loading again
        load_models()

    return HealthResponse(
        status       = "ok" if MODEL["loaded"] else "model_not_loaded",
        model_loaded = MODEL["loaded"],
        n_samples    = MODEL["metadata"]["n_samples"] if MODEL["loaded"] else None,
        cv_accuracy  = MODEL["metadata"]["cv_accuracy_mean"] if MODEL["loaded"] else None,
    )


@app.post("/api/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not MODEL["loaded"]:
        if not load_models():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Run python train.py first."
            )

    # Validate file
    if not file.filename.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    t0 = time.time()

    # Write to temp file and load
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        SR = 22050
        DURATION = 30

        y, sr = librosa.load(tmp_path, sr=SR, duration=DURATION, mono=True)

        target_len = SR * DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))

        # Extract features
        feat = extract_features(y, SR)

        # Scale
        feat_scaled = MODEL["scaler"].transform(feat.reshape(1, -1))

        # LDA project
        lda_coords_arr = MODEL["lda"].transform(feat_scaled)[0]  # (3,)

        # Predict
        proba = MODEL["lda"].predict_proba(feat_scaled)[0]
        pred_idx = int(np.argmax(proba))
        classes = MODEL["le"].classes_

        probabilities = {
            str(cls): float(p)
            for cls, p in zip(classes, proba)
        }

        mood = str(classes[pred_idx])
        confidence = float(proba[pred_idx])

        # Key feature summary (human-readable subset)
        meta = MODEL["metadata"]
        feat_names = meta["feature_names"]
        key_indices = {
            "tempo":            feat_names.index("Tempo"),
            "rms_mean":         feat_names.index("RMS_mean"),
            "zcr_mean":         feat_names.index("ZCR_mean"),
            "spectral_centroid":feat_names.index("Centroid_mean"),
            "harmonic_mean":    feat_names.index("Harmonic_mean"),
            "percussive_mean":  feat_names.index("Percussive_mean"),
            "pitch_mean":       feat_names.index("Pitch_mean"),
            "dynamic_range":    feat_names.index("DynamicRange"),
            "onset_rate":       feat_names.index("OnsetRate"),
            "pulse_clarity":    feat_names.index("PulseClarity"),
        }
        feature_summary = {k: float(feat[v]) for k, v in key_indices.items()}

        elapsed_ms = (time.time() - t0) * 1000

        return PredictResponse(
            mood          = mood,
            confidence    = confidence,
            probabilities = probabilities,
            lda_coords    = {
                "ld1": float(lda_coords_arr[0]),
                "ld2": float(lda_coords_arr[1]),
                "ld3": float(lda_coords_arr[2]),
            },
            features      = feature_summary,
            color         = MOOD_COLORS.get(mood, "#FFFFFF"),
            description   = MOOD_DESCRIPTIONS.get(mood, ""),
            processing_ms = elapsed_ms,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

    finally:
        os.unlink(tmp_path)


@app.get("/api/dataset")
async def dataset():
    """Return full dataset projection for scatter plot."""
    if not MODEL["loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "points":   MODEL["dataset"],
        "centroids": MODEL["metadata"]["centroids"],
        "colors":   MOOD_COLORS,
        "explained_variance_ratio": MODEL["metadata"]["explained_variance_ratio"],
    }


@app.get("/api/discriminants")
async def discriminants():
    """Return LDA discriminant weights for bar charts."""
    if not MODEL["loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded")

    meta = MODEL["metadata"]
    return {
        "axes":     meta["top_features"],
        "colors":   MOOD_COLORS,
        "feature_names": meta["feature_names"],
        "explained_variance_ratio": meta["explained_variance_ratio"],
    }


@app.get("/api/metadata")
async def metadata():
    """Return model metadata."""
    if not MODEL["loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return MODEL["metadata"]


# ─────────────────────────────────────────────────────────────────────────────
# Serve Static Frontend (for production deployment)
# ─────────────────────────────────────────────────────────────────────────────
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).resolve().parent / "static"

if STATIC_DIR.exists():
    # Mount the static directory to serve assets (JS/CSS)
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    # Serve the index.html for the root route and any other unhandled routes (for React Router)
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = STATIC_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(STATIC_DIR / "index.html"))
else:
    print(f"⚠ Static directory not found at {STATIC_DIR}. Frontend will not be served.")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
