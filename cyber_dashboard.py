import streamlit as st
import pandas as pd
import joblib
import time
import random
from datetime import datetime
import shap

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Cyber Threat Dashboard", layout="wide")

# -------------------------------
# Styling
# -------------------------------
st.markdown("""
<style>
.main { background-color: #0e1117; }
h1, h2, h3 { color: #00f5ff; }

[data-testid="metric-container"] {
    background-color: #1c1f26;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #00f5ff;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODELS
# -------------------------------
multi_model = joblib.load("models/xgb_multiclass.pkl")
binary_model = joblib.load("models/xgb_binary.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

explainer = shap.TreeExplainer(multi_model)

data = pd.read_csv("Data/UNSW_NB15_training-set.csv")
data = data.drop(["label", "attack_cat"], axis=1)

# -------------------------------
# Mapping
# -------------------------------
attack_map = {
    0: "Normal", 1: "Analysis", 2: "Backdoor", 3: "DoS",
    4: "Exploits", 5: "Fuzzers", 6: "Generic",
    7: "Reconnaissance", 8: "Shellcode", 9: "Worms"
}

binary_map = {0: "Normal", 1: "Attack"}

severity_map = {
    "Normal": "Low", "Analysis": "Medium", "Backdoor": "High",
    "DoS": "Critical", "Exploits": "Critical", "Fuzzers": "Medium",
    "Generic": "High", "Reconnaissance": "Medium",
    "Shellcode": "Critical", "Worms": "Critical",
    "Attack": "High"
}

mitigation_map = {
    "DoS": "Apply rate limiting and block IP",
    "Exploits": "Patch vulnerabilities",
    "Reconnaissance": "Monitor scanning activity",
    "Fuzzers": "Filter malformed packets",
    "Generic": "Apply firewall rules",
    "Backdoor": "Isolate system",
    "Shellcode": "Inspect and block payload",
    "Worms": "Quarantine system",
    "Analysis": "Monitor closely",
    "Normal": "No action required",
    "Attack": "Investigate traffic"
}

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("⚙️ Control Panel")

run_system = st.sidebar.toggle("Run Detection", value=True)

model_mode = st.sidebar.radio(
    "Select Mode",
    ["Multi-class Detection", "Binary Detection"]
)

attack_filter = st.sidebar.multiselect(
    "Filter Attacks",
    options=list(attack_map.values()) + ["Attack"],
    default=list(attack_map.values())
)

# -------------------------------
# Header
# -------------------------------
st.markdown("""
# 🔐 Real-Time Cyber Threat Detection Dashboard  
### 🚀 ML-powered Intrusion Detection & Risk Monitoring
""")

# -------------------------------
# Session State
# -------------------------------
if "logs" not in st.session_state:
    st.session_state.logs = []
    st.session_state.total_logs = 0
    st.session_state.critical = 0
    st.session_state.high = 0
    st.session_state.normal = 0

# -------------------------------
# DEFAULT VALUES
# -------------------------------
attack = "Normal"
severity = "Low"
mitigation = "No action required"
prediction = 0
shap_df = None   # ✅ STORE SHAP RESULT

# -------------------------------
# Generate Data
# -------------------------------
if run_system:

    sample = data.sample(1).copy()

    if "sbytes" in sample.columns and "dbytes" in sample.columns:
        sample["byte_ratio"] = sample["sbytes"] / (sample["dbytes"] + 1)

    if "sttl" in sample.columns and "dttl" in sample.columns:
        sample["ttl_diff"] = sample["sttl"] - sample["dttl"]

    if "id" not in sample.columns:
        sample["id"] = 0

    for col in sample.select_dtypes(include="object").columns:
        sample[col] = sample[col].astype("category").cat.codes

    sample = sample.reindex(columns=feature_columns, fill_value=0)

    # -------------------------------
    # Realistic Traffic Simulation (FIXED)
    # -------------------------------
    if random.random() < 0.6:
        # Mostly normal traffic
        attack = "Normal"
        severity = "Low"
        mitigation = mitigation_map["Normal"]
        prediction = 0

    else:
        # Use model prediction
        if model_mode == "Multi-class Detection":
            prediction = multi_model.predict(sample)[0]
            attack = attack_map.get(prediction, "Unknown")
        else:
            prediction = binary_model.predict(sample)[0]
            attack = binary_map.get(prediction, "Unknown")

        severity = severity_map.get(attack, "Medium")
        mitigation = mitigation_map.get(attack, "Monitor traffic")


    # ✅ SHAP CALCULATION ONLY (NO UI HERE)
    if model_mode == "Multi-class Detection" and attack != "Normal":
        try:
            shap_values = explainer(sample)
            shap_val = shap_values.values[0][:, int(prediction)]

            shap_df = pd.DataFrame({
                "Feature": sample.columns,
                "Impact": abs(shap_val)
            }).sort_values(by="Impact", ascending=False).head(5).reset_index(drop=True)

        except:
            shap_df = None

    # Metrics
    st.session_state.total_logs += 1

    if severity == "Critical":
        st.session_state.critical += 1
    elif severity == "High":
        st.session_state.high += 1
    elif severity == "Low":
        st.session_state.normal += 1

    log_entry = {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Attack": attack,
        "Severity": severity,
        "Mitigation": mitigation,
        "Mode": model_mode
    }

    st.session_state.logs.append(log_entry)
    st.session_state.logs = st.session_state.logs[-50:]

# -------------------------------
# Metrics
# -------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("📊 Total Logs", st.session_state.total_logs)
col2.metric("🚨 Critical Threats", st.session_state.critical)
col3.metric("⚠️ High Risk", st.session_state.high)
col4.metric("✅ Normal Traffic", st.session_state.normal)

# -------------------------------
# ✅ SHAP DISPLAY (FIXED POSITION)
# -------------------------------
if shap_df is not None:
    st.markdown("### 🔍 Why this prediction?")
    st.dataframe(shap_df, use_container_width=True, hide_index=True)

# -------------------------------
# DataFrame
# -------------------------------
df = pd.DataFrame(st.session_state.logs)

if not df.empty:
    df = df[df["Attack"].isin(attack_filter)]

# -------------------------------
# Status
# -------------------------------
if run_system:
    st.success("🟢 System Running")
else:
    st.warning("🟡 System Paused")

# -------------------------------
# Live Feed
# -------------------------------
st.markdown("## 📡 Live Threat Feed")
st.dataframe(df, use_container_width=True, height=300)

# -------------------------------
# Chart (UNCHANGED LOGIC)
# -------------------------------
st.markdown("## 📊 Attack Distribution")

if not df.empty:
    st.bar_chart(df["Attack"].value_counts())

# -------------------------------
# Alerts
# -------------------------------
if run_system and not df.empty:
    latest = df.iloc[-1]

    if latest["Severity"] == "Critical":
        st.error(f"🚨 CRITICAL ALERT: {latest['Attack']} detected!")
        st.info(f"🛡️ Action: {latest['Mitigation']}")
    elif latest["Severity"] == "High":
        st.warning(f"⚠️ High Risk: {latest['Attack']}")
        st.info(f"🛡️ Action: {latest['Mitigation']}")

# -------------------------------
# Download
# -------------------------------
st.sidebar.download_button(
    label="📥 Download Logs",
    data=df.to_csv(index=False),
    file_name="cyber_logs.csv",
    mime="text/csv"
)

# -------------------------------
# Auto Refresh
# -------------------------------
time.sleep(1)
st.rerun()
