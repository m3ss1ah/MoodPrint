import React, { useState, useCallback, useRef, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import styles from "./UploadZone.module.css";

const WAVEFORM_BARS = 48;

function WaveformLoader({ active, color }) {
  return (
    <div className={styles.waveform} aria-hidden>
      {Array.from({ length: WAVEFORM_BARS }).map((_, i) => (
        <div
          key={i}
          className={styles.bar}
          style={{
            animationDelay: `${(i / WAVEFORM_BARS) * 0.8}s`,
            background: color || "var(--accent)",
            opacity: active ? 1 : 0.3,
          }}
        />
      ))}
    </div>
  );
}

const MOOD_COLORS = {
  Happy:     "#FFD700",
  Energetic: "#FF4500",
  Calm:      "#00CED1",
  Sad:       "#8B7FFF",
};

const STATUS_MSGS = [
  "Loading audio…",
  "Extracting 180 features…",
  "Computing MFCCs…",
  "Analysing harmonic content…",
  "Measuring spectral contrast…",
  "Projecting into LDA space…",
  "Finding nearest mood cluster…",
];

export default function UploadZone({ onPrediction, serverStatus }) {
  const [state, setState]       = useState("idle"); // idle | loading | done | error
  const [fileName, setFileName] = useState(null);
  const [statusMsg, setStatusMsg] = useState("");
  const [msgIdx, setMsgIdx]     = useState(0);
  const [errorMsg, setErrorMsg] = useState("");
  const intervalRef = useRef(null);

  const cycleMessages = () => {
    setMsgIdx(0);
    setStatusMsg(STATUS_MSGS[0]);
    clearInterval(intervalRef.current);
    let i = 1;
    intervalRef.current = setInterval(() => {
      if (i >= STATUS_MSGS.length) { clearInterval(intervalRef.current); return; }
      setStatusMsg(STATUS_MSGS[i]);
      i++;
    }, 600);
  };

  useEffect(() => () => clearInterval(intervalRef.current), []);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setFileName(file.name);
    setState("loading");
    cycleMessages();

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/predict", { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Prediction failed");
      }
      const data = await res.json();
      clearInterval(intervalRef.current);
      setState("done");
      onPrediction(data);
    } catch (e) {
      clearInterval(intervalRef.current);
      setState("error");
      setErrorMsg(e.message);
    }
  }, [onPrediction]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "audio/*": [".mp3", ".wav", ".flac", ".ogg", ".m4a"] },
    multiple: false,
    disabled: serverStatus !== "ready" || state === "loading",
  });

  const offline = serverStatus === "offline";
  const noModel = serverStatus === "no_model";

  return (
    <div className={styles.wrapper}>
      <div className={styles.sectionLabel}>DROP A SONG</div>

      {/* Server warning */}
      {(offline || noModel) && (
        <div className={styles.warning}>
          {offline
            ? "⚠ Backend offline. Start: uvicorn backend.main:app --reload"
            : "⚠ Model not trained. Run: python train.py"
          }
        </div>
      )}

      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`${styles.drop} ${isDragActive ? styles.dragOver : ""} ${
          state === "loading" ? styles.loading : ""
        } ${offline || noModel ? styles.disabled : ""}`}
      >
        <input {...getInputProps()} />

        {state === "idle" && (
          <div className={styles.idleContent}>
            <div className={styles.dropIcon}>
              {isDragActive ? "⬇" : "◈"}
            </div>
            <p className={styles.dropTitle}>
              {isDragActive ? "Release to analyse" : "Drop an audio file"}
            </p>
            <p className={styles.dropSub}>MP3 · WAV · FLAC · OGG · M4A</p>
            <p className={styles.dropHint}>
              First 30 seconds · 180 features extracted
            </p>
          </div>
        )}

        {state === "loading" && (
          <div className={styles.loadingContent}>
            <WaveformLoader active color="var(--accent)" />
            <p className={styles.loadingFile}>{fileName}</p>
            <p className={styles.loadingMsg}>{statusMsg}</p>
          </div>
        )}

        {state === "done" && (
          <div className={styles.doneContent}>
            <WaveformLoader active={false} color="var(--calm)" />
            <p className={styles.doneFile}>{fileName}</p>
            <p className={styles.doneSub}>Analysis complete ↓</p>
            <button
              className={styles.reupload}
              onClick={(e) => { e.stopPropagation(); setState("idle"); setFileName(null); }}
            >
              Upload another
            </button>
          </div>
        )}

        {state === "error" && (
          <div className={styles.errorContent}>
            <div className={styles.errorIcon}>✕</div>
            <p className={styles.errorFile}>{fileName}</p>
            <p className={styles.errorDetail}>{errorMsg}</p>
            <button
              className={styles.reupload}
              onClick={(e) => { e.stopPropagation(); setState("idle"); setFileName(null); setErrorMsg(""); }}
            >
              Try again
            </button>
          </div>
        )}
      </div>

      {/* Feature breakdown */}
      <div className={styles.features}>
        <div className={styles.featLabel}>FEATURE BREAKDOWN</div>
        <div className={styles.featGrid}>
          {[
            ["MFCCs",         "80", "Timbre & texture"],
            ["Chroma",        "24", "Harmonic content"],
            ["Spectral Contrast","14","Peak vs valley energy"],
            ["Tonnetz",       "12", "Harmonic relations"],
            ["Mel Bands",     "24", "Frequency energy"],
            ["Spectral Shape","10", "Centroid, BW, rolloff"],
            ["Rhythm",         "4", "BPM, pulse, onsets"],
            ["Dynamics",       "4", "Range, crest, silence"],
            ["H/P Ratio",      "4", "Harmonic vs percussive"],
            ["Pitch Stats",    "4", "Mean, std, range"],
          ].map(([name, count, desc]) => (
            <div key={name} className={styles.featRow}>
              <span className={styles.featName}>{name}</span>
              <span className={styles.featCount}>{count}</span>
              <span className={styles.featDesc}>{desc}</span>
            </div>
          ))}
          <div className={`${styles.featRow} ${styles.featTotal}`}>
            <span className={styles.featName}>TOTAL</span>
            <span className={styles.featCount}>180</span>
            <span className={styles.featDesc}>→ LDA → 3D</span>
          </div>
        </div>
      </div>
    </div>
  );
}
