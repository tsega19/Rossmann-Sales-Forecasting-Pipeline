import pandas as pd
from logger import configure_logger

logger = configure_logger('data_loader', 'logs/data_loader.log')

def load_data(file_path):
    """Load CSV data from a file path."""
    try:
        logger.info(f"Loading data from {file_path}...")
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise