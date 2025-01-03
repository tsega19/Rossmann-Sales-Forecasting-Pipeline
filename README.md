# Rossmann Sales Forecasting Pipeline

The **Rossmann Sales Forecasting Pipeline** is designed to predict daily store sales by leveraging machine learning techniques. It considers key factors such as promotions, holidays, and competitor proximity to provide accurate sales forecasts up to six weeks in advance. This pipeline supports decision-making by providing real-time predictions via a REST API.

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
   - Evaluates the trained model and generates metrics.
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

### Parameters
The model's parameters are defined in `params.yaml`:
```yaml
model:
  n_estimators: 100
  max_depth: 10
  min_samples_split: 2
  random_state: 42
```

## Key Features
- **Data Cleaning**:
  - Handles missing values, outliers, and ensures data integrity.
- **Feature Engineering**:
  - Includes preprocessing steps for both numerical and categorical data.
- **Visualization**:
  - Distribution plots, correlation heatmaps, and outlier analysis.
- **Evaluation**:
  - Generates metrics to assess model performance.

## Usage

### 1. Install Dependencies
Ensure you have Python and the required libraries installed. Use the following command to install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Pipeline
Run the DVC pipeline stages:
```bash
# Prepare data
dvc repro prepare

# Train model
dvc repro train

# Evaluate model
dvc repro evaluate
```

### 3. Visualize Results
Check evaluation metrics in `metrics.json` and visualize results using notebooks in the `notebook/` folder.

## Future Improvements
- Extend feature engineering to include weather data and regional economic indicators.
- Experiment with deep learning models for improved accuracy.
- Deploy the pipeline using containerized microservices for scalability.

## License
This project is licensed under the MIT License. See `LICENSE` for more details.
