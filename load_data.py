import pandas as pd

train_df = pd.read_csv("data/UNSW_NB15_training-set.csv")
test_df = pd.read_csv("data/UNSW_NB15_testing-set.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Columns:", list(train_df.columns)[:10])
print("Target column check:", train_df[['label', 'attack_cat']].head())
