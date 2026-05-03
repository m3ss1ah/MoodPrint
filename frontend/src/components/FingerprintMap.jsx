import React, { useEffect, useState, useRef } from "react";
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";
import styles from "./FingerprintMap.module.css";

const Plot = createPlotlyComponent(Plotly);

const MOOD_COLORS = {
  Happy:     "#FFD700",
  Energetic: "#FF4500",
  Calm:      "#00CED1",
  Sad:       "#8B7FFF",
};

const MOOD_SYMBOLS = {
  Happy:     "circle",
  Energetic: "diamond",
  Calm:      "square",
  Sad:       "cross",
};

export default function FingerprintMap({ highlight }) {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);
  const [is3D, setIs3D]       = useState(true);
  const [dims, setDims]       = useState({ w: 0, h: 0 });
  const containerRef = useRef(null);

  useEffect(() => {
    fetch("/api/dataset")
      .then((r) => r.json())
      .then(setData)
      .catch(() => setError("Could not load dataset. Is the server running?"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setDims({ w: width, h: height });
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  if (loading) return (
    <div className={styles.center}>
      <div className="spinner" />
      <p className={styles.loadMsg}>Loading fingerprint space…</p>
    </div>
  );

  if (error) return (
    <div className={styles.center}>
      <p className={styles.errorMsg}>⚠ {error}</p>
      <p className={styles.hint}>Run <code>python train.py</code> then restart the backend.</p>
    </div>
  );

  // Group points by mood
  const groups = {};
  for (const pt of data.points) {
    if (!groups[pt.mood]) groups[pt.mood] = { x: [], y: [], z: [], text: [] };
    groups[pt.mood].x.push(pt.ld1);
    groups[pt.mood].y.push(pt.ld2);
    groups[pt.mood].z.push(pt.ld3);
    groups[pt.mood].text.push(pt.file);
  }

  const ev = data.explained_variance_ratio.map((v) => (v * 100).toFixed(1));

  const traces = Object.entries(groups).map(([mood, pts]) => {
    const color = MOOD_COLORS[mood] || "#fff";
    if (is3D) {
      return {
        type: "scatter3d",
        mode: "markers",
        name: mood,
        x: pts.x, y: pts.y, z: pts.z,
        text: pts.text,
        hovertemplate: `<b>${mood}</b><br>%{text}<br>LD1: %{x:.2f} LD2: %{y:.2f} LD3: %{z:.2f}<extra></extra>`,
        marker: {
          size: 4,
          color,
          symbol: MOOD_SYMBOLS[mood],
          opacity: 0.75,
          line: { width: 0 },
        },
      };
    }
    return {
      type: "scatter",
      mode: "markers",
      name: mood,
      x: pts.x, y: pts.y,
      text: pts.text,
      hovertemplate: `<b>${mood}</b><br>%{text}<br>LD1: %{x:.2f} LD2: %{y:.2f}<extra></extra>`,
      marker: {
        size: 7,
        color,
        symbol: MOOD_SYMBOLS[mood],
        opacity: 0.75,
        line: { width: 0.5, color: "rgba(255,255,255,0.15)" },
      },
    };
  });

  // Add centroids
  if (data.centroids) {
    const cx = [], cy = [], cz = [], ctext = [], ccolor = [];
    for (const [mood, coord] of Object.entries(data.centroids)) {
      cx.push(coord[0]);
      cy.push(coord[1]);
      cz.push(coord[2]);
      ctext.push(`⬟ ${mood} centroid`);
      ccolor.push(MOOD_COLORS[mood] || "#fff");
    }
    if (is3D) {
      traces.push({
        type: "scatter3d", mode: "markers+text",
        name: "Centroids",
        x: cx, y: cy, z: cz,
        text: Object.keys(data.centroids),
        textposition: "top center",
        textfont: { color: "#fff", size: 11, family: "Space Mono" },
        marker: { size: 10, color: ccolor, symbol: "diamond", opacity: 1,
                  line: { width: 2, color: "#fff" } },
        hovertemplate: "<b>%{text} centroid</b><extra></extra>",
        showlegend: false,
      });
    }
  }

  // Highlight uploaded song
  if (highlight) {
    const hColor = MOOD_COLORS[highlight.mood] || "#fff";
    if (is3D) {
      traces.push({
        type: "scatter3d", mode: "markers+text",
        name: "Your Song",
        x: [highlight.lda_coords.ld1],
        y: [highlight.lda_coords.ld2],
        z: [highlight.lda_coords.ld3],
        text: ["♪ Your Song"],
        textposition: "top center",
        textfont: { color: "#fff", size: 12, family: "Space Mono" },
        marker: { size: 14, color: hColor, symbol: "circle",
                  line: { width: 3, color: "#fff" }, opacity: 1 },
        hovertemplate: "<b>YOUR SONG</b><br>Mood: " + highlight.mood + "<extra></extra>",
        showlegend: true,
      });
    } else {
      traces.push({
        type: "scatter", mode: "markers+text",
        name: "Your Song",
        x: [highlight.lda_coords.ld1], y: [highlight.lda_coords.ld2],
        text: ["♪"], textposition: "top center",
        textfont: { color: "#fff", size: 14 },
        marker: { size: 18, color: hColor, symbol: "star",
                  line: { width: 2, color: "#fff" }, opacity: 1 },
        hovertemplate: "<b>YOUR SONG</b><br>Mood: " + highlight.mood + "<extra></extra>",
        showlegend: true,
      });
    }
  }

  const plotBg = "rgba(0,0,0,0)";
  const gridColor = "rgba(61,106,255,0.12)";
  const axisColor = "rgba(122,139,181,0.5)";
  const tickColor = "#3d4f75";

  const commonAxis3D = {
    backgroundcolor: "rgba(8,12,20,0.6)",
    gridcolor: gridColor,
    showbackground: true,
    zerolinecolor: axisColor,
    tickfont: { color: tickColor, size: 9, family: "Space Mono" },
    titlefont: { color: axisColor, size: 11, family: "Space Mono" },
  };

  const layout3D = {
    paper_bgcolor: plotBg,
    plot_bgcolor:  plotBg,
    margin: { l: 0, r: 0, t: 20, b: 0 },
    showlegend: true,
    legend: {
      font: { color: "#7a8bb5", size: 11, family: "Space Mono" },
      bgcolor: "rgba(8,12,20,0.8)",
      bordercolor: "#1a2540",
      borderwidth: 1,
      x: 0.01, y: 0.99,
    },
    scene: {
      bgcolor: "rgba(4,5,8,0)",
      xaxis: { ...commonAxis3D, title: `LD1 (${ev[0]}%)` },
      yaxis: { ...commonAxis3D, title: `LD2 (${ev[1]}%)` },
      zaxis: { ...commonAxis3D, title: `LD3 (${ev[2]}%)` },
      camera: { eye: { x: 1.4, y: 1.4, z: 1.0 } },
    },
  };

  const layout2D = {
    paper_bgcolor: plotBg,
    plot_bgcolor:  "rgba(8,12,20,0.4)",
    margin: { l: 60, r: 20, t: 30, b: 60 },
    showlegend: true,
    legend: {
      font: { color: "#7a8bb5", size: 11, family: "Space Mono" },
      bgcolor: "rgba(8,12,20,0.8)",
      bordercolor: "#1a2540",
      borderwidth: 1,
    },
    xaxis: {
      title: { text: `LD1 (${ev[0]}% variance)`, font: { color: axisColor, size: 11, family: "Space Mono" } },
      gridcolor: gridColor,
      zerolinecolor: axisColor,
      tickfont: { color: tickColor, family: "Space Mono", size: 9 },
    },
    yaxis: {
      title: { text: `LD2 (${ev[1]}% variance)`, font: { color: axisColor, size: 11, family: "Space Mono" } },
      gridcolor: gridColor,
      zerolinecolor: axisColor,
      tickfont: { color: tickColor, family: "Space Mono", size: 9 },
    },
  };

  return (
    <div className={styles.container} ref={containerRef}>
      {/* Controls */}
      <div className={styles.toolbar}>
        <div className={styles.info}>
          <span className={styles.infoItem}>
            <span className={styles.infoLabel}>SONGS</span>
            <span className={styles.infoVal}>{data.points.length}</span>
          </span>
          <span className={styles.infoItem}>
            <span className={styles.infoLabel}>VARIANCE</span>
            <span className={styles.infoVal}>{ev.map((v, i) => `LD${i+1}:${v}%`).join(" · ")}</span>
          </span>
        </div>
        <div className={styles.toggles}>
          <button
            className={`${styles.toggle} ${is3D ? styles.toggleActive : ""}`}
            onClick={() => setIs3D(true)}
          >3D</button>
          <button
            className={`${styles.toggle} ${!is3D ? styles.toggleActive : ""}`}
            onClick={() => setIs3D(false)}
          >2D</button>
        </div>
      </div>

      {/* Plot */}
      <div className={styles.plotWrap}>
        <Plot
          data={traces}
          layout={is3D ? layout3D : layout2D}
          config={{
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ["toImage", "sendDataToCloud"],
            responsive: true,
          }}
          style={{ width: "100%", height: "100%" }}
          useResizeHandler
        />
      </div>

      {/* Legend bottom */}
      <div className={styles.moodLegend}>
        {Object.entries(MOOD_COLORS).map(([mood, color]) => (
          <div key={mood} className={styles.moodItem}>
            <span className={styles.moodDot} style={{ background: color }} />
            <span className={styles.moodName}>{mood.toUpperCase()}</span>
          </div>
        ))}
        <div className={styles.modelBadge}>
          Russell's Circumplex · LDA · C−1=3 axes
        </div>
      </div>
    </div>
  );
}
