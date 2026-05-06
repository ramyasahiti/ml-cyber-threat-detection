# train_models.py
#What this does
#Loads your processed data.
#Trains both Random Forest and XGBoost models.
#Prints out performance metrics and confusion matrices.
#Lets you compare results side by side.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load preprocessed data
X_train = pd.read_csv("data/X_train_processed.csv")
X_test = pd.read_csv("data/X_test_processed.csv")
y_train = pd.read_csv("data/y_train_processed.csv").values.ravel()
y_test = pd.read_csv("data/y_test_processed.csv").values.ravel()

# -------------------------
# 1. RANDOM FOREST MODEL
# -------------------------
print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n--- RANDOM FOREST RESULTS ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# -------------------------
# 2. XGBOOST MODEL
# -------------------------
print("\nTraining XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    objective='multi:softmax',  # multi-class classification
    num_class=len(set(y_train))
)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

print("\n--- XGBOOST RESULTS ---")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

print("\n✅ Training and evaluation complete!")
