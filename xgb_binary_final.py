import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("Data/UNSW_NB15_training-set.csv")

# =========================
# BINARY TARGET
# =========================
df["binary_label"] = df["label"]  # already 0 (normal), 1 (attack)

y = df["binary_label"]
X = df.drop(["label", "attack_cat", "binary_label"], axis=1)

# =========================
# ENCODE CATEGORICAL
# =========================
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# =========================
# SAME FEATURE ENGINEERING (IMPORTANT)
# =========================
if "sbytes" in X.columns and "dbytes" in X.columns:
    X["byte_ratio"] = X["sbytes"] / (X["dbytes"] + 1)

if "sttl" in X.columns and "dttl" in X.columns:
    X["ttl_diff"] = X["sttl"] - X["dttl"]

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# MODEL
# =========================
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

print("⏳ Training Binary Model...")
model.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix

# =========================
# EVALUATE
# =========================
y_pred = model.predict(X_test)

print("\n🚀 --- Binary Results ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Labeled version
cm_df = pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

print("\nLabeled Confusion Matrix:\n", cm_df)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=True)
plt.title("Confusion Matrix - Binary Classification")
plt.tight_layout()
plt.show()


# =========================
# SAVE
# =========================
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_binary.pkl")

print("\n💾 Binary model saved!")
