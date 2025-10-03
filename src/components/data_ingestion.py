import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logger


@dataclass
class DataIngestionConfig:
    """Configuration class for data ingestion paths"""
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def read_data_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Read data from CSV file
        Args:
            file_path: Path to the CSV file
        Returns:
            pandas DataFrame
        """
        try:
            logger.info(f"Reading data from CSV: {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        
        except Exception as e:
            logger.error(f"Error reading CSV file: {file_path}")
            raise CustomException(f"Error reading CSV file: {str(e)}", sys.exc_info())
    
    def read_data_from_parquet(self, file_path: str) -> pd.DataFrame:
        """
        Read data from Parquet file
        Args:
            file_path: Path to the Parquet file
        Returns:
            pandas DataFrame
        """
        try:
            logger.info(f"Reading data from Parquet: {file_path}")
            df = pd.read_parquet(file_path)
            logger.info(f"Parquet data loaded successfully. Shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            return df
        
        except Exception as e:
            logger.error(f"Error reading Parquet file: {file_path}")
            raise CustomException(f"Error reading Parquet file: {str(e)}", sys.exc_info())
    
    def read_data_from_file(self, file_path: str) -> pd.DataFrame:
        """
        Generic method to read data from file (auto-detects format)
        Args:
            file_path: Path to the data file
        Returns:
            pandas DataFrame
        """
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.csv':
                return self.read_data_from_csv(file_path)
            elif file_extension == '.parquet':
                return self.read_data_from_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        
        except Exception as e:
            logger.error(f"Error reading file: {file_path}")
            raise CustomException(f"Error reading file: {str(e)}", sys.exc_info())
    
    # TODO: Implement database connections for future use
    def read_data_from_oracle(self, connection_string: str, query: str) -> pd.DataFrame:
        """
        Read data from Oracle database (To be implemented)
        Args:
            connection_string: Oracle database connection string
            query: SQL query to execute
        Returns:
            pandas DataFrame
        """
        # import cx_Oracle  # Uncomment when needed
        # try:
        #     logger.info("Connecting to Oracle database")
        #     connection = cx_Oracle.connect(connection_string)
        #     df = pd.read_sql(query, connection)
        #     connection.close()
        #     logger.info(f"Data loaded from Oracle. Shape: {df.shape}")
        #     return df
        # except Exception as e:
        #     logger.error("Error connecting to Oracle database")
        #     raise CustomException(f"Oracle connection error: {str(e)}", sys.exc_info())
        pass
    
    def read_data_from_mysql(self, connection_params: dict, query: str) -> pd.DataFrame:
        """
        Read data from MySQL database (To be implemented)
        Args:
            connection_params: MySQL connection parameters
            query: SQL query to execute
        Returns:
            pandas DataFrame
        """
        # import mysql.connector  # Uncomment when needed
        # try:
        #     logger.info("Connecting to MySQL database")
        #     connection = mysql.connector.connect(**connection_params)
        #     df = pd.read_sql(query, connection)
        #     connection.close()
        #     logger.info(f"Data loaded from MySQL. Shape: {df.shape}")
        #     return df
        # except Exception as e:
        #     logger.error("Error connecting to MySQL database")
        #     raise CustomException(f"MySQL connection error: {str(e)}", sys.exc_info())
        pass
    
    def create_directories(self) -> None:
        """Create necessary directories for artifacts"""
        try:
            artifacts_dir = Path('artifacts')
            artifacts_dir.mkdir(exist_ok=True)
            logger.info(f"Artifacts directory created: {artifacts_dir}")
        
        except Exception as e:
            logger.error("Error creating directories")
            raise CustomException(f"Directory creation error: {str(e)}", sys.exc_info())
    
    def initiate_data_ingestion(self, source_data_path: str, 
                               test_size: float = 0.2, 
                               random_state: int = 42) -> Tuple[str, str]:
        """
        Main method to initiate data ingestion process
        Args:
            source_data_path: Path to source data file
            test_size: Proportion of test data (default: 0.2)
            random_state: Random seed for reproducibility
        Returns:
            Tuple of (train_data_path, test_data_path)
        """
        logger.info("========== Data Ingestion Started ==========")
        
        try:
            # Create necessary directories
            self.create_directories()
            
            # Read data from source (auto-detects format)
            logger.info("Reading data from source")
            df = self.read_data_from_file(source_data_path)
            
            # Save raw data to artifacts (as CSV for consistency)
            logger.info(f"Saving raw data to: {self.ingestion_config.raw_data_path}")
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            # Initiate train-test split
            logger.info("========== Train Test Split Initiated ==========")
            logger.info(f"Test size: {test_size}, Random state: {random_state}")
            
            train_set, test_set = train_test_split(
                df, 
                test_size=test_size, 
                random_state=random_state,
                stratify=None  # Add stratify column if needed for classification
            )
            
            logger.info(f"Train set shape: {train_set.shape}")
            logger.info(f"Test set shape: {test_set.shape}")
            
            # Save train and test sets
            logger.info(f"Saving train data to: {self.ingestion_config.train_data_path}")
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            
            logger.info(f"Saving test data to: {self.ingestion_config.test_data_path}")
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logger.info("========== Data Ingestion Completed Successfully ==========")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logger.error("Error in data ingestion process")
            raise CustomException(f"Data ingestion failed: {str(e)}", sys.exc_info())


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize data ingestion
        obj = DataIngestion()
        
        # Specify your source data path
        source_path = 'notebooks/data/pivot2_asset_store_3506_Refrigeration_47344.parquet'  # Simplified path
        
        # Run data ingestion
        train_data_path, test_data_path = obj.initiate_data_ingestion(source_path)
        
        logger.info(f"Train data saved at: {train_data_path}")
        logger.info(f"Test data saved at: {test_data_path}")
        
    except CustomException as e:
        logger.error(f"Data ingestion failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")