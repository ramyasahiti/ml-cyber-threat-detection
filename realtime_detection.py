import joblib
import pandas as pd
import time

# Load model and feature columns
model = joblib.load("models/xgb_final_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

print("✅ Model loaded successfully!")

# Load data
data = pd.read_csv("Data/X_test_processed.csv")

# 🔥 ADD THIS HERE
attack_map = {
    0: "Normal",
    1: "Analysis",
    2: "Backdoor",
    3: "DoS",
    4: "Exploits",
    5: "Fuzzers",
    6: "Generic",
    7: "Reconnaissance",
    8: "Shellcode",
    9: "Worms"
}

severity_map = {
    "Normal": "Low",
    "Analysis": "Medium",
    "Backdoor": "High",
    "DoS": "Critical",
    "Exploits": "Critical",
    "Fuzzers": "Medium",
    "Generic": "High",
    "Reconnaissance": "Medium",
    "Shellcode": "Critical",
    "Worms": "Critical"
}


print("🚀 Starting real-time detection...\n")

from datetime import datetime

i = 0

while True:
    sample = data.sample(1)

    # Match training features
    sample = sample[feature_columns]

    prediction = model.predict(sample)[0]

    attack = attack_map.get(prediction, "Unknown")
    severity = severity_map.get(attack, "Unknown")

    current_time = datetime.now().strftime("%H:%M:%S")

    print(f"[{current_time}] Log {i+1}: Attack -> {attack} | Severity -> {severity}")

    # 🚨 Alert system
    if severity == "Critical":
        print("🚨 ALERT: Critical threat detected!\n")

    time.sleep(1)
    i += 1

    # Optional stop (for demo)
    if i == 50:
        print("🛑 Stopping system...")
        break

