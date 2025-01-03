
# Rossmann Sales Forecasting Pipeline

The **Rossmann Sales Forecasting Pipeline** is designed to predict daily store sales by leveraging machine learning techniques. The pipeline considers key factors such as promotions, holidays, and competitor proximity to provide accurate sales forecasts up to six weeks in advance. This system supports decision-making by providing real-time predictions, helping retailers optimize inventory management and marketing strategies.

## Project Structure

```plaintext
Rossmann-Sales-Forecasting-Pipeline/
├── README.md              # Project documentation
├── dvc.lock               # DVC lock file with pipeline stages
├── dvc.yaml               # DVC pipeline definition
├── model.pkl              # Trained model file
├── params.yaml            # Model parameters
├── .dvcignore             # Ignore file for DVC
├── notebook/              # Jupyter notebooks for analysis
│   ├── EDA_preprocessing_test.ipynb
│   ├── EDA_preprocessing_train.ipynb
│   ├── feature_engineering_test.ipynb
│   └── feature_engineering_train.ipynb
├── scripts/               # Python scripts for pipeline
│   ├── data_preprocessing.py
│   ├── feature_Engineering.py
│   ├── logging_utils.py
│   └── utils/
│       ├── data_loader.py
│       └── logger.py
└── .dvc/                  # DVC internal folder
```

## Pipeline Overview

The pipeline follows a series of stages to process data, train the model, and evaluate its performance:

### Stages
1. **Prepare**:
   - Cleans raw data and prepares it for model training.
   - Input: Raw CSV files
   - Output: `data/processed/data.csv`

2. **Train**:
   - Trains a machine learning model on the prepared data.
   - Input: Processed data and parameters
   - Output: `model.pkl`

3. **Evaluate**:
   - Evaluates the trained model and generates performance metrics.
   - Input: Trained model
   - Output: `metrics.json`

### Dependencies
The pipeline uses the following dependencies:
- **Data**:
  - `data/raw/4_test.csv`
  - `data/raw/4_store.csv`
  - `data/raw/4_train.csv`
- **Scripts**:
  - `scripts/prepare.py`
  - `scripts/train.py`
  - `scripts/evaluate.py`

## Key Features
- **Data Cleaning**: Handles missing values, outliers, and ensures data integrity.
- **Feature Engineering**: Includes preprocessing steps for both numerical and categorical data.
- **Visualization**: Distribution plots, correlation heatmaps, and outlier analysis to gain insights into the data.

## Usage

### 1. Install Dependencies
Ensure you have Python and the required libraries installed. Use the following command to install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run Pipeline
Run the DVC pipeline stages to prepare the data, perform feature engineering, and train the model:

```bash
# Prepare data
dvc repro prepare

# Train the model
dvc repro train

# Evaluate the model
dvc repro evaluate
```
