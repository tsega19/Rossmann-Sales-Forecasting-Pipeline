import pandas as pd
import joblib
import yaml
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json

# Load parameters from the params.yaml file
with open('params.yaml', 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# Load the trained model
model = joblib.load('model.pkl')

# Load the processed data for evaluation
data = pd.read_csv('data/processed/data.csv')

# Ensure the same feature engineering steps as during training

# Convert date column to numerical features (if applicable)
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    # Extract features like day, month, year, etc.
    data['day'] = data['Date'].dt.day
    data['month'] = data['Date'].dt.month
    data['year'] = data['Date'].dt.year
    data['weekday'] = data['Date'].dt.weekday
    data.drop('Date', axis=1, inplace=True)  # Drop the original Date column

# Handle non-numeric columns
# Identify columns with non-numeric data
for col in data.select_dtypes(include=['object']).columns:
    # Ensure the column is either all strings or all numbers
    data[col] = data[col].astype(str)  # Convert all values to strings
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Ensure the same columns as during training are used for prediction
# Get the columns that were used during training (excluding 'Sales')
X = data.drop(columns=['Sales'])
y_true = data['Sales']

# Predict with the trained model
predictions = model.predict(X)

# Calculate evaluation metrics
mse = mean_squared_error(y_true, predictions)
rmse = mse ** 0.5
mae = mean_absolute_error(y_true, predictions)

# Save metrics to metrics.json
metrics = {
    "mean_squared_error": mse,
    "root_mean_squared_error": rmse,
    "mean_absolute_error": mae
}

# Write metrics to a JSON file
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

print("Metrics saved to 'metrics.json'")
