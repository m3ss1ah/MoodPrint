"""
train.py — Music Mood LDA Training Pipeline
============================================
Reads sorted audio from data/{Happy,Energetic,Calm,Sad}/
Extracts 180 acoustic features per song using librosa.
Trains a Linear Discriminant Analysis model.
Saves model artefacts to models/.

Feature set (180 total):
  MFCCs           → 40 coefficients × mean+std = 80
  Chroma          → 12 notes × mean+std         = 24
  Spectral Contrast→ 7 bands × mean+std         = 14
  Tonnetz         → 6 × mean+std                = 12
  Mel Spectrogram → 12 bands × mean+std         = 24  (compressed)
  Spectral stats  → centroid, bw, rolloff, zcr, rms × mean+std = 10
  Rhythm          → tempo, beat_strength, onset_rate, pulse_clarity = 4
  Dynamics        → RMS var, dynamic_range, silence_ratio, crest_factor = 4
  Harmonic/Percussive ratio × mean+std = 4
  Pitch stats     → mean, std, median, range = 4
                                              ─────
                                               180

Usage:
    python train.py --data_dir data/ --model_dir models/ --sr 22050 --duration 30
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

CLASSES = ["Happy", "Energetic", "Calm", "Sad"]
FEATURE_DIM = 180


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(file_path: str, sr: int = 22050, duration: int = 30) -> np.ndarray:
    """
    Extract 180 acoustic features from an audio file.
    Returns a 1-D numpy array of length 180 or raises on failure.
    """
    y, sr = librosa.load(file_path, sr=sr, duration=duration, mono=True)

    # Pad if shorter than duration
    target_len = sr * duration
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))

    features = []

    # ── MFCCs (80) ────────────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # ── Chroma STFT (24) ──────────────────────────────────────────────────
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # ── Spectral Contrast (14) ────────────────────────────────────────────
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    # ── Tonnetz (12) ──────────────────────────────────────────────────────
    y_harm = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
    features.extend(np.mean(tonnetz, axis=1))
    features.extend(np.std(tonnetz, axis=1))

    # ── Mel Spectrogram compressed (24) ───────────────────────────────────
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Compress to 12 bands by averaging groups of 10/11 mel bins
    n_bands = 12
    band_size = mel_db.shape[0] // n_bands
    mel_bands = np.array([
        mel_db[i * band_size:(i + 1) * band_size].mean(axis=0)
        for i in range(n_bands)
    ])
    features.extend(np.mean(mel_bands, axis=1))
    features.extend(np.std(mel_bands, axis=1))

    # ── Spectral shape stats (10) ─────────────────────────────────────────
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr       = librosa.feature.zero_crossing_rate(y)[0]
    rms       = librosa.feature.rms(y=y)[0]

    for feat in [centroid, bandwidth, rolloff, zcr, rms]:
        features.append(np.mean(feat))
        features.append(np.std(feat))

    # ── Rhythm features (4) ───────────────────────────────────────────────
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    onset_env    = librosa.onset.onset_strength(y=y, sr=sr)
    onset_rate   = len(librosa.onset.onset_detect(y=y, sr=sr)) / (len(y) / sr)
    pulse        = librosa.beat.plp(onset_envelope=onset_env, sr=sr)

    features.append(float(tempo))
    features.append(float(np.mean(onset_env)))
    features.append(float(onset_rate))
    features.append(float(np.mean(pulse)))

    # ── Dynamic features (4) ──────────────────────────────────────────────
    rms_arr      = librosa.feature.rms(y=y)[0]
    rms_var      = float(np.var(rms_arr))
    dynamic_rng  = float(np.max(rms_arr) - np.min(rms_arr))
    silence_mask = rms_arr < 1e-4
    silence_ratio= float(np.mean(silence_mask))
    peak_rms     = float(np.max(np.abs(y)))
    crest_factor = float(peak_rms / (np.mean(rms_arr) + 1e-9))

    features.extend([rms_var, dynamic_rng, silence_ratio, crest_factor])

    # ── Harmonic / Percussive ratio (4) ───────────────────────────────────
    y_harm, y_perc = librosa.effects.hpss(y)
    h_rms = librosa.feature.rms(y=y_harm)[0]
    p_rms = librosa.feature.rms(y=y_perc)[0]
    features.append(float(np.mean(h_rms)))
    features.append(float(np.std(h_rms)))
    features.append(float(np.mean(p_rms)))
    features.append(float(np.std(p_rms)))

    # ── Pitch stats (4) ───────────────────────────────────────────────────
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[magnitudes > np.percentile(magnitudes, 75)]
    pitch_vals  = pitch_vals[pitch_vals > 0]
    if len(pitch_vals) > 0:
        features.extend([
            float(np.mean(pitch_vals)),
            float(np.std(pitch_vals)),
            float(np.median(pitch_vals)),
            float(np.ptp(pitch_vals)),
        ])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])

    feat_arr = np.array(features, dtype=np.float32)

    # Sanity check
    assert len(feat_arr) == FEATURE_DIM, \
        f"Feature count mismatch: got {len(feat_arr)}, expected {FEATURE_DIM}"

    # Replace NaN/Inf
    feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return feat_arr


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(data_dir: str, sr: int, duration: int):
    data_path = Path(data_dir)
    X, y, paths = [], [], []

    total_files = sum(
        len(list((data_path / cls).glob("*.mp3")))
        for cls in CLASSES
        if (data_path / cls).exists()
    )

    print(f"\n🎵 Extracting features from {total_files} files...\n")
    processed = 0
    errors = 0
    t0 = time.time()

    for cls in CLASSES:
        cls_dir = data_path / cls
        if not cls_dir.exists():
            print(f"⚠  Class folder not found: {cls_dir}")
            continue

        files = list(cls_dir.glob("*.mp3")) + list(cls_dir.glob("*.wav"))
        print(f"  📂 {cls:<12} → {len(files)} files")

        for fpath in files:
            try:
                feat = extract_features(str(fpath), sr=sr, duration=duration)
                X.append(feat)
                y.append(cls)
                paths.append(str(fpath))
                processed += 1

                elapsed = time.time() - t0
                rate = processed / elapsed
                remaining = (total_files - processed) / rate if rate > 0 else 0
                print(
                    f"\r  Progress: {processed}/{total_files} | "
                    f"{rate:.1f} files/s | ETA: {remaining:.0f}s    ",
                    end="", flush=True
                )
            except Exception as e:
                errors += 1
                print(f"\n  ✗ Error on {fpath.name}: {e}")

    print(f"\n\n✅ Done: {processed} features extracted, {errors} errors")
    print(f"⏱  Total time: {time.time() - t0:.1f}s")

    return np.array(X), np.array(y), paths


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(data_dir: str, model_dir: str, sr: int, duration: int):
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y, paths = load_dataset(data_dir, sr, duration)

    if len(X) == 0:
        raise RuntimeError(
            "No audio files found. "
            "Run prepare_deam.py first to populate data/ folders."
        )

    print(f"\n📐 Feature matrix: {X.shape}  (samples × features)")
    print(f"📊 Class distribution:")
    for cls in CLASSES:
        n = np.sum(y == cls)
        print(f"  {cls:<12} {n} samples")

    # Encode labels
    le = LabelEncoder()
    le.fit(CLASSES)
    y_enc = le.transform(y)

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── LDA ────────────────────────────────────────────────────────────────
    # n_components = C-1 = 3 (we have 4 classes)
    # solver='svd' is numerically stable, no matrix inversion needed
    # store_covariance=True so we can compute Mahalanobis later
    lda = LinearDiscriminantAnalysis(
        n_components=3,
        solver="svd",
        store_covariance=True,
        tol=1e-6,
    )
    lda.fit(X_scaled, y_enc)

    print(f"\n🧠 LDA trained:")
    print(f"   Discriminant axes: {lda.n_components}")
    ev = lda.explained_variance_ratio_
    for i, v in enumerate(ev):
        print(f"   LD{i+1}: {v*100:.1f}% variance explained")
    print(f"   Total: {sum(ev)*100:.1f}%")

    # ── Cross-validation ───────────────────────────────────────────────────
    print("\n📈 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(lda, X_scaled, y_enc, cv=cv, scoring="accuracy")
    print(f"   CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"   Per-fold:    {' | '.join(f'{s:.3f}' for s in cv_scores)}")

    # Full-dataset fit report
    y_pred = lda.predict(X_scaled)
    print("\n📋 Classification report (train set):")
    print(classification_report(y_enc, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_enc, y_pred)
    print("Confusion matrix:")
    print(cm)

    # ── Project all samples for scatter plot ───────────────────────────────
    X_lda = lda.transform(X_scaled)   # shape: (N, 3)

    # Build per-class centroid in LDA space
    centroids = {}
    for cls_idx, cls_name in enumerate(le.classes_):
        mask = y_enc == cls_idx
        centroids[cls_name] = X_lda[mask].mean(axis=0).tolist()

    # ── Feature importance ─────────────────────────────────────────────────
    # scalings_ shape: (n_features, n_components)
    scalings = lda.scalings_   # (180, 3)

    # Generate feature names
    feature_names = (
        [f"MFCC_{i+1}_mean"    for i in range(40)] +
        [f"MFCC_{i+1}_std"     for i in range(40)] +
        [f"Chroma_{i+1}_mean"  for i in range(12)] +
        [f"Chroma_{i+1}_std"   for i in range(12)] +
        [f"SpContrast_{i+1}_mean" for i in range(7)] +
        [f"SpContrast_{i+1}_std"  for i in range(7)] +
        [f"Tonnetz_{i+1}_mean" for i in range(6)] +
        [f"Tonnetz_{i+1}_std"  for i in range(6)] +
        [f"Mel_{i+1}_mean"     for i in range(12)] +
        [f"Mel_{i+1}_std"      for i in range(12)] +
        ["Centroid_mean", "Centroid_std",
         "Bandwidth_mean", "Bandwidth_std",
         "Rolloff_mean", "Rolloff_std",
         "ZCR_mean", "ZCR_std",
         "RMS_mean", "RMS_std"] +
        ["Tempo", "OnsetStrength", "OnsetRate", "PulseClarity"] +
        ["RMS_var", "DynamicRange", "SilenceRatio", "CrestFactor"] +
        ["Harmonic_mean", "Harmonic_std", "Percussive_mean", "Percussive_std"] +
        ["Pitch_mean", "Pitch_std", "Pitch_median", "Pitch_range"]
    )
    assert len(feature_names) == FEATURE_DIM, \
        f"Feature name count: {len(feature_names)} vs {FEATURE_DIM}"

    # Top 20 features for each LD axis
    top_features = {}
    for axis in range(3):
        weights = scalings[:, axis]
        abs_w = np.abs(weights)
        top_idx = np.argsort(abs_w)[::-1][:20]
        top_features[f"LD{axis+1}"] = [
            {"feature": feature_names[i], "weight": float(weights[i])}
            for i in top_idx
        ]

    # ── Save artefacts ─────────────────────────────────────────────────────
    joblib.dump(lda,    model_path / "lda.pkl")
    joblib.dump(scaler, model_path / "scaler.pkl")
    joblib.dump(le,     model_path / "label_encoder.pkl")

    # Save dataset projection for frontend scatter plot
    dataset_projection = []
    for i in range(len(X_lda)):
        dataset_projection.append({
            "ld1": float(X_lda[i, 0]),
            "ld2": float(X_lda[i, 1]),
            "ld3": float(X_lda[i, 2]),
            "mood": y[i],
            "file": Path(paths[i]).name,
        })

    metadata = {
        "classes":              CLASSES,
        "n_features":           FEATURE_DIM,
        "n_components":         3,
        "feature_names":        feature_names,
        "explained_variance_ratio": [float(v) for v in lda.explained_variance_ratio_],
        "cv_accuracy_mean":     float(cv_scores.mean()),
        "cv_accuracy_std":      float(cv_scores.std()),
        "centroids":            centroids,
        "top_features":         top_features,
        "scalings_shape":       list(scalings.shape),
        "n_samples":            int(len(X)),
        "class_counts": {
            cls: int(np.sum(y == cls)) for cls in CLASSES
        },
    }

    with open(model_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(model_path / "dataset_projection.json", "w") as f:
        json.dump(dataset_projection, f)

    print(f"\n💾 Saved to {model_path.resolve()}:")
    print(f"   lda.pkl, scaler.pkl, label_encoder.pkl")
    print(f"   metadata.json, dataset_projection.json")
    print(f"\n🎉 Training complete! CV accuracy: {cv_scores.mean()*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LDA mood classifier on DEAM")
    parser.add_argument("--data_dir",   default="data",   help="Root of mood-sorted audio")
    parser.add_argument("--model_dir",  default="models", help="Where to save artefacts")
    parser.add_argument("--sr",         default=22050, type=int, help="Sample rate")
    parser.add_argument("--duration",   default=30,    type=int, help="Seconds per clip")
    args = parser.parse_args()

    train(args.data_dir, args.model_dir, args.sr, args.duration)
