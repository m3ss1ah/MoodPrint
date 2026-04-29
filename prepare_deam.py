"""
prepare_deam.py — DEAM Dataset Preprocessor
============================================
Converts DEAM's continuous valence/arousal annotations
into 4 discrete mood classes using Russell's Circumplex Model:

    High Valence + High Arousal  →  Happy
    Low  Valence + High Arousal  →  Energetic
    High Valence + Low  Arousal  →  Calm
    Low  Valence + Low  Arousal  →  Sad

DEAM directory structure after downloading from Kaggle:
    deam/
        MEMD_audio/          ← .mp3 files named by song ID (e.g. 2.mp3)
        annotations/
            static_annotations_averaged_songs_1_2000.csv

Usage:
    python prepare_deam.py --deam_dir deam/ --output_dir data/

This will copy audio files into:
    data/
        Happy/
        Energetic/
        Calm/
        Sad/
"""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
import numpy as np


MOOD_MAP = {
    (True,  True):  "Happy",
    (False, True):  "Energetic",
    (True,  False): "Calm",
    (False, False): "Sad",
}


def prepare(deam_dir: str, output_dir: str):
    deam = Path(deam_dir)
    out  = Path(output_dir)

    # ── Find annotation CSVs ──────────────────────────────────────────────
    csv_candidates = sorted(deam.rglob("static_annotations_averaged_songs*.csv"))
    if not csv_candidates:
        csv_candidates = sorted(deam.rglob("*.csv"))

    if not csv_candidates:
        raise FileNotFoundError(
            f"No annotation CSV found in {deam_dir}. "
            "Expected: static_annotations_averaged_songs_1_2000.csv"
        )

    # Merge all annotation CSVs (they may have different column schemas)
    dfs = []
    for csv_path in csv_candidates:
        print(f"📄 Reading annotations: {csv_path}")
        tmp = pd.read_csv(csv_path)
        tmp.columns = [c.strip().lower().replace(" ", "_") for c in tmp.columns]
        print(f"   Columns: {tmp.columns.tolist()}")
        print(f"   Shape: {tmp.shape}")

        # Extract only the columns we need
        id_col = next((c for c in tmp.columns if c in ("song_id", "songid", "id")), tmp.columns[0])
        val_col = next((c for c in tmp.columns if "valence" in c and "mean" in c and "max" not in c and "min" not in c), None)
        aro_col = next((c for c in tmp.columns if "arousal" in c and "mean" in c and "max" not in c and "min" not in c), None)

        if val_col is None:
            val_col = next((c for c in tmp.columns if "valence" in c), None)
        if aro_col is None:
            aro_col = next((c for c in tmp.columns if "arousal" in c), None)

        if val_col and aro_col:
            subset = tmp[[id_col, val_col, aro_col]].copy()
            subset.columns = ["song_id", "valence", "arousal"]
            dfs.append(subset)
            print(f"   → Using: id='{id_col}', valence='{val_col}', arousal='{aro_col}'")
        else:
            print(f"   ⚠ Skipping {csv_path.name}: missing valence/arousal columns")

    if not dfs:
        raise FileNotFoundError("No usable annotation CSV found with valence/arousal columns.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"\n📊 Combined annotations: {df.shape[0]} songs")

    df = df[["song_id", "valence", "arousal"]].dropna()
    df["song_id"] = df["song_id"].astype(int)

    # ── Bin into quadrants ────────────────────────────────────────────────
    v_med = df["valence"].median()
    a_med = df["arousal"].median()
    print(f"\n📊 Valence median: {v_med:.2f}  |  Arousal median: {a_med:.2f}")

    df["high_valence"] = df["valence"] >= v_med
    df["high_arousal"] = df["arousal"] >= a_med
    df["mood"] = df.apply(
        lambda r: MOOD_MAP[(r["high_valence"], r["high_arousal"])], axis=1
    )

    print("\n📦 Class distribution:")
    print(df["mood"].value_counts().to_string())

    # ── Find audio files ──────────────────────────────────────────────────
    audio_dir = deam / "MEMD_audio"
    if not audio_dir.exists():
        mp3_folders = set(p.parent for p in deam.rglob("*.mp3"))
        if not mp3_folders:
            raise FileNotFoundError(
                f"No .mp3 files found under {deam_dir}. "
                "Make sure you downloaded the MEMD_audio folder."
            )
        audio_dir = list(mp3_folders)[0]
        print(f"⚠  Using audio dir: {audio_dir}")

    # ── Copy files into mood folders ──────────────────────────────────────
    for mood in MOOD_MAP.values():
        (out / mood).mkdir(parents=True, exist_ok=True)

    copied, missing = 0, 0
    for _, row in df.iterrows():
        song_id = int(row["song_id"])
        mood    = row["mood"]

        candidates = [
            audio_dir / f"{song_id}.mp3",
            audio_dir / f"{song_id:04d}.mp3",
        ]
        src = next((p for p in candidates if p.exists()), None)

        if src is None:
            missing += 1
            continue

        dst = out / mood / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        copied += 1

    print(f"\n✅ Copied {copied} files  |  ⚠  Missing {missing} files")
    print(f"📁 Output: {out.resolve()}")

    print("\nFinal counts per class:")
    for mood in MOOD_MAP.values():
        n = len(list((out / mood).glob("*.mp3")))
        print(f"  {mood:<12} {n} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deam_dir",   default="deam",  help="Path to downloaded DEAM folder")
    parser.add_argument("--output_dir", default="data",  help="Output path for mood-sorted audio")
    args = parser.parse_args()
    prepare(args.deam_dir, args.output_dir)
