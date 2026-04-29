import React, { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import styles from "./DiscriminantPanel.module.css";

export default function DiscriminantPanel() {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);
  const [activeAxis, setActiveAxis] = useState("LD1");

  useEffect(() => {
    fetch("/api/discriminants")
      .then((r) => r.json())
      .then(setData)
      .catch(() => setError("Could not load discriminant data."))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className={styles.center}>
      <div className="spinner" />
      <p className={styles.msg}>Loading discriminant weights…</p>
    </div>
  );

  if (error) return (
    <div className={styles.center}>
      <p className={styles.err}>{error}</p>
    </div>
  );

  const ev = data.explained_variance_ratio;
  const axisData = data.axes[activeAxis] || [];

  // Sort by absolute weight
  const sorted = [...axisData].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
  const top = sorted.slice(0, 20);

  const barColors = top.map((d) =>
    d.weight > 0
      ? `rgba(61,106,255,0.85)`
      : `rgba(255,100,60,0.85)`
  );

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor:  "rgba(8,12,20,0.4)",
    margin: { l: 180, r: 60, t: 20, b: 50 },
    xaxis: {
      title: {
        text: "Discriminant Weight",
        font: { color: "#7a8bb5", size: 11, family: "Space Mono" },
      },
      gridcolor: "rgba(61,106,255,0.1)",
      zerolinecolor: "rgba(122,139,181,0.6)",
      zerolinewidth: 1.5,
      tickfont: { color: "#3d4f75", family: "Space Mono", size: 9 },
    },
    yaxis: {
      tickfont: { color: "#7a8bb5", family: "Space Mono", size: 10 },
      autorange: "reversed",
    },
    showlegend: false,
    hovermode: "y unified",
    hoverlabel: {
      bgcolor: "#0d1220",
      bordercolor: "#2a3f6f",
      font: { color: "#e8eeff", family: "Space Mono", size: 11 },
    },
  };

  const traces = [{
    type: "bar",
    orientation: "h",
    y: top.map((d) => d.feature),
    x: top.map((d) => d.weight),
    marker: {
      color: barColors,
      line: { width: 0 },
    },
    hovertemplate: "<b>%{y}</b><br>Weight: %{x:.4f}<extra></extra>",
  }];

  // Category groupings for annotation
  const CATEGORY_MAP = {
    MFCC: "#3d6aff",
    Chroma: "#00ced1",
    SpContrast: "#ffd700",
    Tonnetz: "#8b7fff",
    Mel: "#ff8c00",
    Centroid: "#00fa9a",
    Bandwidth: "#00fa9a",
    Rolloff: "#00fa9a",
    ZCR: "#00fa9a",
    RMS: "#00fa9a",
    Tempo: "#ff69b4",
    OnsetStrength: "#ff69b4",
    OnsetRate: "#ff69b4",
    PulseClarity: "#ff69b4",
    DynamicRange: "#ff4500",
    Harmonic: "#7cfc00",
    Percussive: "#7cfc00",
    Pitch: "#da70d6",
  };

  function getCategoryColor(featureName) {
    for (const [prefix, color] of Object.entries(CATEGORY_MAP)) {
      if (featureName.startsWith(prefix)) return color;
    }
    return "#7a8bb5";
  }

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div>
          <div className={styles.title}>Discriminant Weights</div>
          <div className={styles.subtitle}>
            Top 20 features driving each LDA axis.
            Positive = pushes toward cluster boundary. Negative = opposing direction.
          </div>
        </div>

        {/* Axis selector */}
        <div className={styles.axisSel}>
          {["LD1", "LD2", "LD3"].map((ax, i) => (
            <button
              key={ax}
              className={`${styles.axBtn} ${activeAxis === ax ? styles.axActive : ""}`}
              onClick={() => setActiveAxis(ax)}
            >
              <span className={styles.axName}>{ax}</span>
              <span className={styles.axEv}>{(ev[i] * 100).toFixed(1)}%</span>
            </button>
          ))}
        </div>
      </div>

      <div className={styles.body}>
        {/* Bar chart */}
        <div className={styles.chartWrap}>
          <Plot
            data={traces}
            layout={layout}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%", height: "100%" }}
            useResizeHandler
          />
        </div>

        {/* Right panel: legend + explanation */}
        <div className={styles.side}>
          <div className={styles.sideSection}>
            <div className={styles.sideLabel}>FEATURE GROUPS</div>
            {[
              ["MFCC 1-40",    "#3d6aff",  "Timbre texture"],
              ["Chroma 1-12",  "#00ced1",  "Harmonic content"],
              ["SpContrast",   "#ffd700",  "Peak/valley energy"],
              ["Tonnetz",      "#8b7fff",  "Harmonic relations"],
              ["Mel Bands",    "#ff8c00",  "Mel spectrogram"],
              ["Spectral",     "#00fa9a",  "Shape stats"],
              ["Rhythm",       "#ff69b4",  "Beat & onset"],
              ["Dynamics",     "#ff4500",  "RMS range, crest"],
              ["Harm/Perc",    "#7cfc00",  "Source separation"],
              ["Pitch",        "#da70d6",  "Pitch distribution"],
            ].map(([name, color, desc]) => (
              <div key={name} className={styles.legendRow}>
                <span className={styles.legendDot} style={{ background: color }} />
                <span className={styles.legendName}>{name}</span>
                <span className={styles.legendDesc}>{desc}</span>
              </div>
            ))}
          </div>

          <div className={styles.sideSection}>
            <div className={styles.sideLabel}>HOW TO READ</div>
            <div className={styles.explainBlock}>
              <div className={styles.explainItem}>
                <span className={styles.posBar} />
                <span className={styles.explainText}>
                  Positive weight → feature increases along this axis
                </span>
              </div>
              <div className={styles.explainItem}>
                <span className={styles.negBar} />
                <span className={styles.explainText}>
                  Negative weight → feature decreases along this axis
                </span>
              </div>
              <p className={styles.explainNote}>
                LDA maximises between-class variance ÷ within-class variance.
                Features with large |weight| are the most discriminative for
                separating the 4 mood classes in this axis direction.
              </p>
            </div>
          </div>

          <div className={styles.sideSection}>
            <div className={styles.sideLabel}>VARIANCE EXPLAINED</div>
            {["LD1", "LD2", "LD3"].map((ax, i) => (
              <div key={ax} className={styles.evRow}>
                <span className={styles.evAxis}>{ax}</span>
                <div className={styles.evBarTrack}>
                  <div
                    className={styles.evBarFill}
                    style={{ width: `${ev[i] * 100}%` }}
                  />
                </div>
                <span className={styles.evPct}>{(ev[i] * 100).toFixed(1)}%</span>
              </div>
            ))}
            <div className={styles.evTotal}>
              Total: {(ev.reduce((a, b) => a + b, 0) * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
