"""
Streamlit dashboard for the food rescue prediction service.

Shows:
- Service health (polls /health)
- Drift status (runs KS test against training distribution)
- Action distribution of live predictions
- Latency over time
- Recent prediction log

Run with:
    streamlit run monitoring/dashboard.py
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # ensure project root on path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Food Rescue — ML Monitor",
    page_icon="🥗",
    layout="wide",
)

st.title("🥗 Food Rescue RL — Prediction Monitor")

API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000")
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto-refresh every 10s", value=False)

if auto_refresh:
    import time
    time.sleep(10)
    st.rerun()

# ------------------------------------------------------------------
# 1. Service health
# ------------------------------------------------------------------
st.header("Service Health")
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    info = requests.get(f"{API_URL}/info", timeout=3).json()
    metrics = requests.get(f"{API_URL}/metrics", timeout=3).json()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", health["status"].upper())
    col2.metric("Model", f"{info['model_name']} v{info['model_version']}")
    col3.metric("Total Predictions", metrics["total_predictions"])
    avg_lat = metrics.get("avg_latency_ms")
    col4.metric("Avg Latency", f"{avg_lat:.1f} ms" if avg_lat else "—")

except Exception as e:
    st.error(f"Cannot reach API at {API_URL}: {e}")
    st.stop()

# ------------------------------------------------------------------
# 2. Drift detection
# ------------------------------------------------------------------
st.header("Distribution Drift")

if st.button("Run Drift Check"):
    with st.spinner("Running KS test..."):
        try:
            from monitoring.drift_detector import DriftDetector
            detector = DriftDetector()
            report = detector.run()

            if report.drift_detected:
                st.error(f"⚠️ {report.summary()}")
            else:
                st.success(f"✅ {report.summary()}")

            if report.n_live >= 30:
                pval_df = pd.DataFrame({
                    "feature_index": list(range(len(report.feature_pvalues))),
                    "p_value": report.feature_pvalues,
                    "drifted": [p < report.threshold for p in report.feature_pvalues],
                })
                st.dataframe(pval_df, use_container_width=True)

        except Exception as e:
            st.error(f"Drift check failed: {e}")

# ------------------------------------------------------------------
# 3. Prediction log
# ------------------------------------------------------------------
st.header("Recent Predictions")

try:
    from api.prediction_log import fetch_recent
    rows = fetch_recent(200)

    if not rows:
        st.info("No predictions logged yet. Send some requests to /predict first.")
    else:
        df = pd.DataFrame(rows)
        df["timestamp_iso"] = pd.to_datetime(df["timestamp_iso"])

        # Action distribution
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Action Distribution")
            action_counts = df["action_kind"].value_counts().reset_index()
            action_counts.columns = ["action_kind", "count"]
            st.bar_chart(action_counts.set_index("action_kind"))

        with col_b:
            st.subheader("Latency Over Time (ms)")
            latency_df = df[["timestamp_iso", "latency_ms"]].sort_values("timestamp_iso")
            st.line_chart(latency_df.set_index("timestamp_iso"))

        st.subheader("Log Table")
        display_cols = ["request_id", "timestamp_iso", "action", "action_kind",
                        "model_name", "latency_ms"]
        st.dataframe(df[display_cols].head(50), use_container_width=True)

except Exception as e:
    st.error(f"Could not load prediction log: {e}")
