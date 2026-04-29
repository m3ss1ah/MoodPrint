import React from "react";
import styles from "./ResultPanel.module.css";

const MOOD_COLORS = {
  Happy:     "#FFD700",
  Energetic: "#FF4500",
  Calm:      "#00CED1",
  Sad:       "#8B7FFF",
};

const MOOD_ICONS = {
  Happy:     "◆",
  Energetic: "▲",
  Calm:      "■",
  Sad:       "●",
};

const FEATURE_LABELS = {
  tempo:             "Tempo (BPM)",
  rms_mean:          "RMS Energy",
  zcr_mean:          "Zero Crossing Rate",
  spectral_centroid: "Spectral Centroid (Hz)",
  harmonic_mean:     "Harmonic Energy",
  percussive_mean:   "Percussive Energy",
  pitch_mean:        "Mean Pitch (Hz)",
  dynamic_range:     "Dynamic Range",
  onset_rate:        "Onset Rate (/s)",
  pulse_clarity:     "Pulse Clarity",
};

function round(v, n = 3) {
  return typeof v === "number" ? v.toFixed(n) : v;
}

export default function ResultPanel({ result, onViewOnMap }) {
  const { mood, confidence, probabilities, lda_coords, features,
          color, description, processing_ms } = result;

  const moodColor = MOOD_COLORS[mood] || color;

  return (
    <div className={styles.wrapper}>
      <div className={styles.sectionLabel}>RESULT</div>

      {/* Mood card */}
      <div
        className={styles.moodCard}
        style={{
          borderColor: moodColor,
          boxShadow: `0 0 30px ${moodColor}22, 0 0 60px ${moodColor}11`,
        }}
      >
        <div className={styles.moodIcon} style={{ color: moodColor }}>
          {MOOD_ICONS[mood] || "◈"}
        </div>
        <div className={styles.moodName} style={{ color: moodColor }}>
          {mood.toUpperCase()}
        </div>
        <div className={styles.moodDesc}>{description}</div>
        <div className={styles.confidence}>
          <span className={styles.confLabel}>CONFIDENCE</span>
          <span className={styles.confVal} style={{ color: moodColor }}>
            {(confidence * 100).toFixed(1)}%
          </span>
        </div>

        {/* Confidence bar */}
        <div className={styles.confBar}>
          <div
            className={styles.confFill}
            style={{ width: `${confidence * 100}%`, background: moodColor }}
          />
        </div>

        <button className={styles.mapBtn} onClick={onViewOnMap}
          style={{ borderColor: moodColor, color: moodColor }}>
          View on Fingerprint Map →
        </button>
      </div>

      {/* Probability bars */}
      <div className={styles.probSection}>
        <div className={styles.subLabel}>CLASS PROBABILITIES</div>
        {Object.entries(probabilities)
          .sort(([, a], [, b]) => b - a)
          .map(([cls, prob]) => (
            <div key={cls} className={styles.probRow}>
              <span className={styles.probLabel}
                style={{ color: MOOD_COLORS[cls] || "#fff" }}>
                {cls}
              </span>
              <div className={styles.probBarTrack}>
                <div
                  className={styles.probBarFill}
                  style={{
                    width: `${prob * 100}%`,
                    background: MOOD_COLORS[cls] || "var(--accent)",
                    opacity: cls === mood ? 1 : 0.45,
                  }}
                />
              </div>
              <span className={styles.probPct}>
                {(prob * 100).toFixed(1)}%
              </span>
            </div>
          ))}
      </div>

      {/* LDA coordinates */}
      <div className={styles.coordSection}>
        <div className={styles.subLabel}>LDA COORDINATES</div>
        <div className={styles.coordGrid}>
          {["ld1", "ld2", "ld3"].map((axis, i) => (
            <div key={axis} className={styles.coordCard}>
              <span className={styles.coordAxis}>LD{i + 1}</span>
              <span className={styles.coordVal}>{round(lda_coords[axis], 3)}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Key features */}
      <div className={styles.featSection}>
        <div className={styles.subLabel}>KEY FEATURES</div>
        <div className={styles.featList}>
          {Object.entries(features).map(([key, val]) => (
            <div key={key} className={styles.featItem}>
              <span className={styles.featKey}>{FEATURE_LABELS[key] || key}</span>
              <span className={styles.featVal}>{round(val, 4)}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className={styles.footer}>
        <span>Processed in {processing_ms.toFixed(0)} ms</span>
        <span>180 features → LDA(3)</span>
      </div>
    </div>
  );
}
