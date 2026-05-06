# preprocess_data.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the raw datasets
train_df = pd.read_csv("data/UNSW_NB15_training-set.csv")
test_df = pd.read_csv("data/UNSW_NB15_testing-set.csv")

# Drop non-useful columns
cols_to_drop = ['id','srcip','sport','dstip','dsport']
train_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
test_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Identify categorical columns
cat_cols = ['proto','service','state']

# Encode categorical columns
for col in cat_cols:
    le = LabelEncoder()
    le.fit(train_df[col])

    # Transform training data
    train_df[col] = le.transform(train_df[col])

    # Transform test data — handle unseen labels
    test_df[col] = test_df[col].map(lambda s: -1 if s not in le.classes_ else le.transform([s])[0])


# Encode target column (attack_cat)
le_target = LabelEncoder()
train_df['attack_cat'] = le_target.fit_transform(train_df['attack_cat'])
test_df['attack_cat']  = le_target.transform(test_df['attack_cat'])

# Separate features (X) and labels (y)
X_train = train_df.drop(['label','attack_cat'], axis=1)
y_train = train_df['attack_cat']
X_test  = test_df.drop(['label','attack_cat'], axis=1)
y_test  = test_df['attack_cat']

# Scale numeric features (optional but helps)
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Save processed data
X_train.to_csv("data/X_train_processed.csv", index=False)
X_test.to_csv("data/X_test_processed.csv", index=False)
y_train.to_csv("data/y_train_processed.csv", index=False)
y_test.to_csv("data/y_test_processed.csv", index=False)

print("✅ Preprocessing complete. Processed files saved in /data folder.")
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
