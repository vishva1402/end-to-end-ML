import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation paths and parameters
    Using dataclass for type safety and automatic method generation
    """
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    
    # Feature engineering parameters based on your XGBoost notebook insights
    outlier_threshold: float = 2.5  # Z-score threshold for outlier detection
    cv_threshold: float = 0.2       # Coefficient of variation threshold
    rolling_window: int = 24        # 24-hour rolling window for stability
    
    # Imputation strategies
    numerical_strategy: str = 'median'  # Less sensitive to outliers
    categorical_strategy: str = 'most_frequent'
    
    # Scaling method
    scaling_method: str = 'robust'  # Better for data with outliers


class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for advanced feature engineering based on your XGBoost insights
    Implements the same feature creation logic from your notebook
    """
    
    def __init__(self, create_lag_features: bool = True, create_rolling_features: bool = True):
        self.create_lag_features = create_lag_features
        self.create_rolling_features = create_rolling_features
        
    def fit(self, X, y=None):
        """Fit method - learn any necessary parameters"""
        logger.info("Fitting AdvancedFeatureEngineer")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform method - apply feature engineering"""
        try:
            logger.info("Starting advanced feature engineering transformation")
            X_transformed = X.copy()
            
            # Sort by datetime if available for time-based features
            if 'datetime' in X_transformed.columns:
                X_transformed = X_transformed.sort_values('datetime').reset_index(drop=True)
            
            # 1. TIME-BASED FEATURES (from your notebook)
            if 'datetime' in X_transformed.columns:
                X_transformed['datetime'] = pd.to_datetime(X_transformed['datetime'])
                X_transformed['hour_of_day'] = X_transformed['datetime'].dt.hour
                X_transformed['day_of_week'] = X_transformed['datetime'].dt.dayofweek
                X_transformed['month'] = X_transformed['datetime'].dt.month
                X_transformed['is_weekend'] = X_transformed['day_of_week'].isin([5, 6]).astype(int)
                X_transformed['is_business_hours'] = X_transformed['hour_of_day'].between(8, 18).astype(int)
                
                # Cyclic encoding for circular features
                X_transformed['hour_sin'] = np.sin(2 * np.pi * X_transformed['hour_of_day'] / 24)
                X_transformed['hour_cos'] = np.cos(2 * np.pi * X_transformed['hour_of_day'] / 24)
                
                logger.info("Created time-based features")
            
            # 2. TEMPERATURE FEATURES (from your notebook)
            if 'weatherProfileType_temperature' in X_transformed.columns:
                # Cooling degree days (base 18°C for refrigeration)
                X_transformed['cooling_degree_days'] = np.maximum(0, X_transformed['weatherProfileType_temperature'] - 18)
                
                # Temperature lags if we have enough data
                if len(X_transformed) > 24:
                    X_transformed['temperature_lag_24h'] = X_transformed['weatherProfileType_temperature'].shift(24)
                    X_transformed['temperature_change_24h'] = (X_transformed['weatherProfileType_temperature'] - 
                                                              X_transformed['temperature_lag_24h'])
                
                logger.info("Created temperature-based features")
            
            # 3. CONSUMPTION LAG FEATURES (if target variable exists)
            if 'total_consumption' in X_transformed.columns and self.create_lag_features and len(X_transformed) > 72:
                for lag in [1, 24, 48]:
                    X_transformed[f'consumption_lag_{lag}h'] = X_transformed['total_consumption'].shift(lag)
                
                logger.info("Created consumption lag features")
            
            # 4. ROLLING STATISTICS (from your notebook)
            if 'total_consumption' in X_transformed.columns and self.create_rolling_features and len(X_transformed) > 168:
                # Rolling means
                X_transformed['consumption_rolling_24h_mean'] = (X_transformed['total_consumption']
                                                               .shift(1).rolling(window=24, min_periods=12).mean())
                X_transformed['consumption_rolling_72h_mean'] = (X_transformed['total_consumption']
                                                               .shift(1).rolling(window=72, min_periods=36).mean())
                
                # Rolling statistics for stability detection
                X_transformed['consumption_rolling_std'] = (X_transformed['total_consumption']
                                                           .rolling(window=24, min_periods=12).std())
                X_transformed['consumption_rolling_mean'] = (X_transformed['total_consumption']
                                                            .rolling(window=24, min_periods=12).mean())
                X_transformed['consumption_cv'] = (X_transformed['consumption_rolling_std'] / 
                                                  np.maximum(0.1, X_transformed['consumption_rolling_mean']))
                
                logger.info("Created rolling statistical features")
            
            # 5. SUBMETER FEATURES (if available - from your notebook)
            submeter_cols = [col for col in X_transformed.columns if 'submeter' in col]
            if submeter_cols:
                # Basic submeter statistics
                if 'active_submeter_count' in X_transformed.columns:
                    total_submeters = X_transformed.get('total_submeter_count', 10)  # Default assumption
                    X_transformed['active_submeter_ratio'] = (X_transformed['active_submeter_count'] / 
                                                             np.maximum(1, total_submeters))
                
                if 'submeter_consumption_max' in X_transformed.columns and 'total_consumption' in X_transformed.columns:
                    X_transformed['dominant_submeter_ratio'] = (X_transformed['submeter_consumption_max'] / 
                                                               np.maximum(0.1, X_transformed['total_consumption']))
                
                logger.info("Created submeter-based features")
            
            # 6. INTERACTION FEATURES (key insights from your analysis)
            if all(col in X_transformed.columns for col in ['weatherProfileType_temperature', 'hour_of_day']):
                X_transformed['temp_hour_interaction'] = (X_transformed['weatherProfileType_temperature'] * 
                                                         X_transformed['hour_of_day'])
            
            if all(col in X_transformed.columns for col in ['cooling_degree_days', 'is_weekend']):
                X_transformed['cooling_weekend_interaction'] = (X_transformed['cooling_degree_days'] * 
                                                               X_transformed['is_weekend'])
            
            # 7. STABILITY INDICATORS (from your regime detection analysis)
            if 'consumption_cv' in X_transformed.columns:
                X_transformed['is_stable_period'] = (X_transformed['consumption_cv'] <= 0.2).astype(int)
            
            logger.info(f"Advanced feature engineering complete. Created {X_transformed.shape[1] - X.shape[1]} new features")
            
            return X_transformed
            
        except Exception as e:
            logger.error("Error in advanced feature engineering")
            raise CustomException(f"Feature engineering failed: {str(e)}", sys.exc_info())


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Custom transformer for outlier detection and handling based on your notebook insights
    Uses the same outlier detection logic from your stability analysis
    """
    
    def __init__(self, method: str = 'zscore', threshold: float = 2.5, strategy: str = 'cap'):
        self.method = method
        self.threshold = threshold
        self.strategy = strategy  # 'cap', 'remove', or 'flag'
        self.outlier_bounds_ = {}
        
    def fit(self, X, y=None):
        """Learn outlier bounds from training data"""
        logger.info(f"Fitting OutlierHandler with method={self.method}, threshold={self.threshold}")
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if self.method == 'zscore':
                mean = X[col].mean()
                std = X[col].std()
                lower_bound = mean - self.threshold * std
                upper_bound = mean + self.threshold * std
            elif self.method == 'iqr':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.threshold * IQR
                upper_bound = Q3 + self.threshold * IQR
            
            self.outlier_bounds_[col] = {'lower': lower_bound, 'upper': upper_bound}
        
        return self
    
    def transform(self, X):
        """Apply outlier handling"""
        X_transformed = X.copy()
        outlier_count = 0
        
        for col, bounds in self.outlier_bounds_.items():
            if col in X_transformed.columns:
                outliers = ((X_transformed[col] < bounds['lower']) | 
                           (X_transformed[col] > bounds['upper']))
                outlier_count += outliers.sum()
                
                if self.strategy == 'cap':
                    X_transformed[col] = np.clip(X_transformed[col], bounds['lower'], bounds['upper'])
                elif self.strategy == 'flag':
                    X_transformed[f'{col}_is_outlier'] = outliers.astype(int)
        
        logger.info(f"Handled {outlier_count} outliers using {self.strategy} strategy")
        return X_transformed


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Create the preprocessing pipeline based on your notebook insights
        Uses robust scaling and advanced imputation strategies
        """
        try:
            logger.info("Creating data transformation pipeline")
            
            # Define feature categories based on your analysis
            numerical_features = [
                'weatherProfileType_temperature', 'cooling_degree_days',
                'active_submeter_count', 'submeter_consumption_max', 'submeter_consumption_mean',
                'submeter_consumption_std', 'hour_of_day', 'day_of_week', 'month',
                'consumption_lag_24h', 'consumption_rolling_24h_mean', 'consumption_cv',
                'dominant_submeter_ratio', 'temp_hour_interaction'
            ]
            
            categorical_features = [
                'is_weekend', 'is_business_hours', 'is_stable_period'
            ]
            
            # Numerical pipeline with advanced preprocessing
            numerical_pipeline = Pipeline(steps=[
                ('outlier_handler', OutlierHandler(
                    method='zscore',
                    threshold=self.transformation_config.outlier_threshold,
                    strategy='cap'
                )),
                ('imputer', KNNImputer(n_neighbors=5)),  # More sophisticated imputation
                ('scaler', RobustScaler() if self.transformation_config.scaling_method == 'robust' 
                          else StandardScaler())
            ])
            
            # Categorical pipeline
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=self.transformation_config.categorical_strategy)),
                ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ])
            
            # Combine pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical', numerical_pipeline, numerical_features),
                    ('categorical', categorical_pipeline, categorical_features)
                ],
                remainder='passthrough',  # Keep other columns as-is
                sparse_threshold=0  # Return dense array
            )
            
            logger.info("Data transformation pipeline created successfully")
            logger.info(f"Numerical features: {len(numerical_features)}")
            logger.info(f"Categorical features: {len(categorical_features)}")
            
            return preprocessor
            
        except Exception as e:
            logger.error("Error creating data transformer")
            raise CustomException(f"Data transformer creation failed: {str(e)}", sys.exc_info())
    
    def initiate_data_transformation(self, train_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Main method to initiate data transformation process
        Applies the same preprocessing logic from your notebook
        """
        try:
            logger.info("========== Data Transformation Started ==========")
            
            # Read train and test data
            logger.info(f"Reading train data from: {train_path}")
            train_df = pd.read_csv(train_path)
            
            logger.info(f"Reading test data from: {test_path}")
            test_df = pd.read_csv(test_path)
            
            logger.info(f"Train data shape: {train_df.shape}")
            logger.info(f"Test data shape: {test_df.shape}")
            
            # Identify target column (based on your notebook)
            target_column_name = "total_consumption"
            
            if target_column_name not in train_df.columns:
                raise CustomException(f"Target column '{target_column_name}' not found in training data", sys.exc_info())
            
            # Apply advanced feature engineering
            logger.info("========== Advanced Feature Engineering Started ==========")
            feature_engineer = AdvancedFeatureEngineer(
                create_lag_features=True,
                create_rolling_features=True
            )
            
            # Fit and transform training data
            train_df_engineered = feature_engineer.fit_transform(train_df)
            test_df_engineered = feature_engineer.transform(test_df)
            
            logger.info(f"After feature engineering - Train shape: {train_df_engineered.shape}")
            logger.info(f"After feature engineering - Test shape: {test_df_engineered.shape}")
            
            # Separate features and target
            input_feature_train_df = train_df_engineered.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df_engineered[target_column_name]
            
            input_feature_test_df = test_df_engineered.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df_engineered[target_column_name]
            
            # Create and fit preprocessing pipeline
            logger.info("========== Preprocessing Pipeline Started ==========")
            preprocessing_obj = self.get_data_transformer_object()
            
            # Fit on training data only (prevent data leakage)
            logger.info("Fitting preprocessing pipeline on training data")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            
            # Transform test data using fitted pipeline
            logger.info("Transforming test data using fitted pipeline")
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # Combine features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logger.info(f"Final train array shape: {train_arr.shape}")
            logger.info(f"Final test array shape: {test_arr.shape}")
            
            # Save preprocessing object
            logger.info(f"Saving preprocessing object to: {self.transformation_config.preprocessor_obj_file_path}")
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            # Data quality checks (based on your stability analysis)
            self._perform_data_quality_checks(train_arr, test_arr)
            
            logger.info("========== Data Transformation Completed Successfully ==========")
            
            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logger.error("Error in data transformation process")
            raise CustomException(f"Data transformation failed: {str(e)}", sys.exc_info())
    
    def _perform_data_quality_checks(self, train_arr: np.ndarray, test_arr: np.ndarray) -> None:
        """
        Perform data quality checks based on your notebook insights
        """
        try:
            logger.info("========== Data Quality Checks Started ==========")
            
            # Check for NaN values
            train_nan_count = np.isnan(train_arr).sum()
            test_nan_count = np.isnan(test_arr).sum()
            
            if train_nan_count > 0:
                logger.warning(f"Training data contains {train_nan_count} NaN values")
            
            if test_nan_count > 0:
                logger.warning(f"Test data contains {test_nan_count} NaN values")
            
            # Check for infinite values
            train_inf_count = np.isinf(train_arr).sum()
            test_inf_count = np.isinf(test_arr).sum()
            
            if train_inf_count > 0:
                logger.warning(f"Training data contains {train_inf_count} infinite values")
            
            if test_inf_count > 0:
                logger.warning(f"Test data contains {test_inf_count} infinite values")
            
            # Check target variable distribution (last column)
            train_target = train_arr[:, -1]
            test_target = test_arr[:, -1]
            
            logger.info(f"Training target - Mean: {train_target.mean():.2f}, Std: {train_target.std():.2f}")
            logger.info(f"Test target - Mean: {test_target.mean():.2f}, Std: {test_target.std():.2f}")
            
            # Check for data leakage (target distributions should be similar)
            target_correlation = np.corrcoef(
                np.histogram(train_target, bins=50)[0],
                np.histogram(test_target, bins=50)[0]
            )[0, 1]
            
            if target_correlation < 0.7:
                logger.warning(f"Low correlation between train/test target distributions: {target_correlation:.3f}")
            else:
                logger.info(f"Good train/test target correlation: {target_correlation:.3f}")
            
            logger.info("========== Data Quality Checks Completed ==========")
            
        except Exception as e:
            logger.warning(f"Data quality checks failed: {str(e)}")


if __name__ == "__main__":
    # Example usage
    try:
        obj = DataTransformation()
        
        # Specify paths to your train and test data
        train_data_path = "artifacts/train.csv"
        test_data_path = "artifacts/test.csv"
        
        # Run data transformation
        train_array, test_array, preprocessor_path = obj.initiate_data_transformation(
            train_data_path, test_data_path
        )
        
        logger.info(f"Transformation complete!")
        logger.info(f"Train array shape: {train_array.shape}")
        logger.info(f"Test array shape: {test_array.shape}")
        logger.info(f"Preprocessor saved at: {preprocessor_path}")
        
    except CustomException as e:
        logger.error(f"Data transformation failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")