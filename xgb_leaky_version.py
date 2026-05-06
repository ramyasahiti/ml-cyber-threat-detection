import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("Data/UNSW_NB15_training-set.csv")

# Drop unnecessary columns
df = df.drop(columns=["id"], errors="ignore")

# Encode categorical columns
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Target column
y = df["label"]
X = df.drop(columns=["label"])

print(f"✅ Loaded data with {X.shape[1]} features")

# =========================
# OPTIONAL: MERGE RARE CLASSES (SAFE VERSION)
# =========================
def merge_rare_classes(y, threshold=1000):
    counts = y.value_counts()
    rare_classes = counts[counts < threshold].index

    y = y.copy()
    y[y.isin(rare_classes)] = 10  # new merged class

    return y

y = merge_rare_classes(y)

# =========================
# RELABEL (VERY IMPORTANT - FIXES YOUR ERROR)
# =========================
def relabel_classes(y):
    unique_classes = sorted(y.unique())
    mapping = {old: new for new, old in enumerate(unique_classes)}
    return y.map(mapping)

y = relabel_classes(y)

print("✅ Classes relabeled properly")

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# MODEL (YOUR BEST SETTINGS)
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
    tree_method="hist",   # faster
    n_jobs=-1
)

print("⏳ Training Final XGBoost Model...")
model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("\n--- 🚀 Final Results ---")
print(f"Accuracy: {acc:.4f}")

print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_final_model.pkl")

print("\n💾 Model saved successfully!")
print("✅ DONE")



