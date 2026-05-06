# feature_selection_retrain.py
#The results will now directly reflect the updated, post–feature-selection metrics.
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Load processed data
# ---------------------------------------------------------
X_train = pd.read_csv("data/X_train_processed.csv")
X_test = pd.read_csv("data/X_test_processed.csv")
y_train = pd.read_csv("data/y_train_processed.csv").values.ravel()
y_test = pd.read_csv("data/y_test_processed.csv").values.ravel()

# ---------------------------------------------------------
# 1. Get Feature Importance from XGBoost
# ---------------------------------------------------------
print("Training initial XGBoost to get feature importance...")
xgb_temp = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    objective="multi:softmax",
    num_class=len(set(y_train))
)
xgb_temp.fit(X_train, y_train)

# Get feature importance
importances = pd.Series(xgb_temp.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=False)

# Plot top 20 features
plt.figure(figsize=(10,5))
importances.head(20).plot(kind="bar")
plt.title("Top 20 Feature Importances (XGBoost)")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 2. Select top 25 most important features
# ---------------------------------------------------------
top_features = importances.head(25).index.tolist()
X_train_top = X_train[top_features]
X_test_top  = X_test[top_features]

# ---------------------------------------------------------
# 3. Retrain Random Forest and XGBoost on selected features
# ---------------------------------------------------------
# ---------------------------------------------------------
# 3. Retrain Random Forest and XGBoost on selected features
# ---------------------------------------------------------
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

print("\nRetraining Random Forest with top 25 features...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_top, y_train)
rf_preds = rf_model.predict(X_test_top)

# ---- RF Metrics ----
rf_acc = accuracy_score(y_test, rf_preds)
rf_prec, rf_rec, rf_f1, _ = precision_recall_fscore_support(y_test, rf_preds, average='weighted')
print("\n--- RANDOM FOREST RESULTS (After Feature Selection) ---")
print(f"Accuracy: {rf_acc:.4f}")
print(f"Weighted Precision: {rf_prec:.4f}")
print(f"Weighted Recall: {rf_rec:.4f}")
print(f"Weighted F1-score: {rf_f1:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, rf_preds))
print("\nDetailed Report:\n", classification_report(y_test, rf_preds))

print("\nRetraining XGBoost with top 25 features...")
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
    objective="multi:softmax",
    num_class=len(set(y_train))
)
xgb_model.fit(X_train_top, y_train)
xgb_preds = xgb_model.predict(X_test_top)

# ---- XGB Metrics ----
xgb_acc = accuracy_score(y_test, xgb_preds)
xgb_prec, xgb_rec, xgb_f1, _ = precision_recall_fscore_support(y_test, xgb_preds, average='weighted')
print("\n--- XGBOOST RESULTS (After Feature Selection) ---")
print(f"Accuracy: {xgb_acc:.4f}")
print(f"Weighted Precision: {xgb_prec:.4f}")
print(f"Weighted Recall: {xgb_rec:.4f}")
print(f"Weighted F1-score: {xgb_f1:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, xgb_preds))
print("\nDetailed Report:\n", classification_report(y_test, xgb_preds))

print("\n✅ Feature selection and retraining complete!")

