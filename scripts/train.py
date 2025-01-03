import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import yaml

# Load parameters from the params.yaml file
with open('params.yaml', 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# Access model parameters
n_estimators = params['model']['n_estimators']
max_depth = params['model']['max_depth']
min_samples_split = params['model']['min_samples_split']
random_state = params['model']['random_state']

# Load processed data
data = pd.read_csv('data/processed/data.csv')

# Check for non-numeric values in the data
non_numeric_cols = data.select_dtypes(exclude=['number']).columns
print("Non-numeric columns:", non_numeric_cols)

# Check the rows that contain non-numeric values
print(data[non_numeric_cols].applymap(lambda x: isinstance(x, str)).head())

# Handle non-numeric columns by converting or removing them
for col in non_numeric_cols:
    if data[col].dtype == 'object':
        # If the column is categorical, apply LabelEncoder
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
    else:
        # If the column contains unexpected non-numeric values (like 'a'), clean it
        data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert errors to NaN
        data[col].fillna(0, inplace=True)  # Replace NaNs with 0 or another value

# Convert date column to numerical features (if applicable)
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    # Extract features like day, month, year, etc.
    data['day'] = data['Date'].dt.day
    data['month'] = data['Date'].dt.month
    data['year'] = data['Date'].dt.year
    data['weekday'] = data['Date'].dt.weekday
    data.drop('Date', axis=1, inplace=True)  # Drop the original Date column

# Separate features and target
X = data.drop(columns=['Sales'])
y = data['Sales']

# Split into train and validation sets (optional, but good practice)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Train model (using Random Forest as an example)
model = RandomForestRegressor(n_estimators=n_estimators,
                              max_depth=max_depth,
                              min_samples_split=min_samples_split,
                              random_state=random_state)

model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'model.pkl')
