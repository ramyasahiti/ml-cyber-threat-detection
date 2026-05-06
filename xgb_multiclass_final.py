import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("Data/UNSW_NB15_training-set.csv")

print("✅ Loaded data with", df.shape[1] - 2, "features")

# =========================
# CORRECT TARGET (MULTI-CLASS)
# =========================
y = df["attack_cat"]   # 🔥 IMPORTANT FIX
X = df.drop(["label", "attack_cat"], axis=1)

# =========================
# ENCODE TARGET
# =========================
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Save encoder for Streamlit later
os.makedirs("models", exist_ok=True)
joblib.dump(target_encoder, "models/target_encoder.pkl")

# =========================
# ENCODE CATEGORICAL FEATURES
# =========================
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# =========================
# FEATURE ENGINEERING (SAFE)
# =========================
if "sbytes" in X.columns and "dbytes" in X.columns:
    X["byte_ratio"] = X["sbytes"] / (X["dbytes"] + 1)

if "sttl" in X.columns and "dttl" in X.columns:
    X["ttl_diff"] = X["sttl"] - X["dttl"]

print("✅ Features after engineering:", X.shape[1])

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# MODEL (STABLE SETTINGS)
# =========================
model = XGBClassifier(
    n_estimators=250,
    max_depth=6,
    learning_rate=0.07,
    subsample=0.9,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=len(np.unique(y)),
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

print("⏳ Training XGBoost Multi-class Model...")
model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)

print("\n🚀 --- Final Multi-class Results ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, average="weighted"), 4))
print("Recall:", round(recall_score(y_test, y_pred, average="weighted"), 4))
print("F1-Score:", round(f1_score(y_test, y_pred, average="weighted"), 4))

print("\nDetailed Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Labeled confusion matrix
class_names = target_encoder.classes_

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {c}" for c in class_names],
    columns=[f"Predicted {c}" for c in class_names]
)

print("\nLabeled Confusion Matrix:\n", cm_df)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Get class names from your encoder
import joblib
enc = joblib.load("models/target_encoder.pkl")
class_names = enc.classes_

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap="Blues", colorbar=True)
plt.title("Confusion Matrix - Cyber Threat Detection")
plt.xticks(rotation=45, ha='right')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
# =========================
# SAVE MODEL + FEATURES
# =========================
joblib.dump(model, "models/xgb_multiclass.pkl")
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

print("\n💾 Model + encoder saved successfully!")
print("✅ DONE (REAL MULTI-CLASS MODEL)")
