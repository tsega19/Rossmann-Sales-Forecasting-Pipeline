import pandas as pd
import os

# Load raw data
train_data = pd.read_csv('data/raw/4_train.csv')
store_data = pd.read_csv('data/raw/4_store.csv')
test_data = pd.read_csv('data/raw/4_test.csv')

# Merge the data or perform necessary cleaning/preprocessing
merged_data = pd.merge(train_data, store_data, on='Store')

# Ensure that the 'data/processed' directory exists
os.makedirs('data/processed', exist_ok=True)

# Save the processed data
merged_data.to_csv('data/processed/data.csv', index=False)
