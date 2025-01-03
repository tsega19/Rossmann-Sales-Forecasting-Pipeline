import logging
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
import sys
sys.path.append(os.path.abspath('../scripts'))
from utils.logger import configure_logger

def build_pipeline():
    """
    Build an ML pipeline with preprocessing and a RandomForestRegressor.
    """
    # logger.info("Building ML pipeline.")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    # logger.info("Pipeline built successfully.")
    return pipeline

def train_model(pipeline, X_train, y_train):
    """
    Train the ML model.
    """
    # logger.info("Training model.")
    pipeline.fit(X_train, y_train)
    # logger.info("Model trained successfully.")
    return pipeline

def save_model(model, file_name):
    """
    Save the trained model.
    """
    # logger.info("Saving model to %s", file_name)
    joblib.dump(model, file_name)
    # logger.info("Model saved.")
