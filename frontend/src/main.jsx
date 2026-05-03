import React from "react";
import ReactDOM from "react-dom/client";
import { ErrorBoundary } from "react-error-boundary";
import App from "./App";
import "./styles/global.css";

// Polyfill for plotly/vite
if (typeof window !== "undefined") {
  window.global = window;
}

function ErrorFallback({ error }) {
  return (
    <div style={{ color: "red", padding: "20px", background: "#111", height: "100vh", fontFamily: "monospace" }}>
      <h2>FATAL REACT ERROR</h2>
      <pre style={{ whiteSpace: "pre-wrap" }}>{error.message}</pre>
      <pre style={{ whiteSpace: "pre-wrap", marginTop: "10px", fontSize: "12px", color: "#f88" }}>{error.stack}</pre>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);
