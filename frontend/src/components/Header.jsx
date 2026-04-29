import React, { useRef, useState, useEffect } from "react";
import styles from "./Header.module.css";

export default function Header({ tabs, activeTab, onTabChange }) {
  const tabRefs = useRef({});
  const [indicatorStyle, setIndicatorStyle] = useState({});

  useEffect(() => {
    const el = tabRefs.current[activeTab];
    if (el) {
      setIndicatorStyle({
        left:  el.offsetLeft + "px",
        width: el.offsetWidth + "px",
      });
    }
  }, [activeTab]);

  return (
    <header className={styles.header}>
      {/* Logo */}
      <div className={styles.logo}>
        <span className={styles.logoIcon}>◈</span>
        <span className={styles.logoText}>MOODPRINT</span>
        <span className={styles.logoSub}>LDA · DEAM · Russell's Model</span>
      </div>

      {/* Tabs */}
      <nav className={styles.nav}>
        <div className={styles.indicator} style={indicatorStyle} />
        {tabs.map((tab) => (
          <button
            key={tab.id}
            ref={(el) => (tabRefs.current[tab.id] = el)}
            className={`${styles.tab} ${activeTab === tab.id ? styles.active : ""}`}
            onClick={() => onTabChange(tab.id)}
          >
            <span className={styles.tabShort}>{tab.short}</span>
            <span className={styles.tabFull}>{tab.label}</span>
          </button>
        ))}
      </nav>

      {/* Right badge */}
      <div className={styles.badge}>
        <span className={styles.badgeDot} />
        <span>180 FEATURES</span>
      </div>
    </header>
  );
}
