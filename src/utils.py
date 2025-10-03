import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from src.exception import CustomException
from src.logger import logger


def save_object(file_path: str, obj) -> None:
    """
    Save a Python object to file using pickle
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logger.info(f"Object saved successfully to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving object to {file_path}")
        raise CustomException(f"Error saving object: {str(e)}", sys.exc_info())


def load_object(file_path: str):
    """
    Load a Python object from file using pickle
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        
        logger.info(f"Object loaded successfully from {file_path}")
        return obj
        
    except Exception as e:
        logger.error(f"Error loading object from {file_path}")
        raise CustomException(f"Error loading object: {str(e)}", sys.exc_info())


def evaluate_models(X_train, y_train, X_test, y_test, models: dict) -> dict:
    """
    Evaluate multiple models and return performance metrics
    """
    try:
        model_report = {}
        
        for model_name, model in models.items():
            logger.info(f"Training {model_name}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics
            from sklearn.metrics import r2_score
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)
            
            model_report[model_name] = {
                'train_score': train_score,
                'test_score': test_score
            }
            
            logger.info(f"{model_name} - Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
        
        return model_report
        
    except Exception as e:
        logger.error("Error in model evaluation")
        raise CustomException(f"Model evaluation failed: {str(e)}", sys.exc_info())