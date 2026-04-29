import React, { useState, useEffect } from "react";
import styles from "./App.module.css";
import Header from "./components/Header";
import StatusBar from "./components/StatusBar";
import FingerprintMap from "./components/FingerprintMap";
import UploadZone from "./components/UploadZone";
import ResultPanel from "./components/ResultPanel";
import DiscriminantPanel from "./components/DiscriminantPanel";

const TABS = [
  { id: "map",          label: "Fingerprint Map",     short: "MAP" },
  { id: "upload",       label: "Drop a Song",          short: "UPLOAD" },
  { id: "discriminants",label: "Discriminant Weights", short: "WEIGHTS" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("map");
  const [serverStatus, setServerStatus] = useState("checking");
  const [prediction, setPrediction] = useState(null);
  const [mapHighlight, setMapHighlight] = useState(null);

  // Poll server health
  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch("/api/health");
        const data = await res.json();
        setServerStatus(data.model_loaded ? "ready" : "no_model");
      } catch {
        setServerStatus("offline");
      }
    };
    check();
    const id = setInterval(check, 15000);
    return () => clearInterval(id);
  }, []);

  const handlePrediction = (result) => {
    setPrediction(result);
    setMapHighlight(result);
    setActiveTab("upload");
  };

  return (
    <div className={styles.root}>
      <Header
        tabs={TABS}
        activeTab={activeTab}
        onTabChange={setActiveTab}
      />

      <main className={styles.content}>
        {activeTab === "map" && (
          <FingerprintMap highlight={mapHighlight} />
        )}
        {activeTab === "upload" && (
          <div className={styles.uploadLayout}>
            <UploadZone
              onPrediction={handlePrediction}
              serverStatus={serverStatus}
            />
            {prediction && (
              <ResultPanel
                result={prediction}
                onViewOnMap={() => setActiveTab("map")}
              />
            )}
          </div>
        )}
        {activeTab === "discriminants" && (
          <DiscriminantPanel />
        )}
      </main>

      <StatusBar status={serverStatus} />
    </div>
  );
}
