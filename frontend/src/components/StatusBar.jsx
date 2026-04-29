import React from "react";
import styles from "./StatusBar.module.css";

const STATUS_CONFIG = {
  checking: { dot: "checking", text: "Connecting to backend…" },
  ready:    { dot: "ready",    text: "Model loaded · Ready" },
  no_model: { dot: "warn",     text: "Run python train.py to train model" },
  offline:  { dot: "offline",  text: "Backend offline · uvicorn backend.main:app --reload" },
};

export default function StatusBar({ status }) {
  const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.checking;

  return (
    <footer className={styles.bar}>
      <div className={styles.left}>
        <span className={`${styles.dot} ${styles[cfg.dot]}`} />
        <span className={styles.text}>{cfg.text}</span>
      </div>
      <div className={styles.right}>
        <span className={styles.item}>DEAM · 2058 songs</span>
        <span className={styles.sep}>·</span>
        <span className={styles.item}>Russell's Circumplex Model</span>
        <span className={styles.sep}>·</span>
        <span className={styles.item}>sklearn LDA · 180 features</span>
      </div>
    </footer>
  );
}
