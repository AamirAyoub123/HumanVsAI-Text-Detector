
import pandas as pd
from pathlib import Path
import sys
import os
current_file_path = Path(__file__).resolve()
PROJECT_ROOT = current_file_path.parent.parent.parent 
sys.path.append(str(PROJECT_ROOT))
from src.utils.paths import TEST_DATA_PATH, VALIDATION_DATA_PATH, PROCESSED_DATA_DIR
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_test_data(test_df: pd.DataFrame, validation_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
 
    if not 0 <= validation_frac <= 1:
        raise ValueError("validation_frac must be between 0 and 1")
        
    if test_df.empty:
        logger.warning("Received empty DataFrame for splitting")
        return pd.DataFrame(), pd.DataFrame()

    logger.info(f"Splitting test data with validation fraction: {validation_frac}")
    
    # Shuffle the test DataFrame
    test_df = test_df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split index
    validation_size = int(len(test_df) * validation_frac)

    # Split the DataFrame
    validation_df = test_df.iloc[:validation_size]
    new_test_df = test_df.iloc[validation_size:]
    
    # Save the split datasets
    validation_df.to_csv(VALIDATION_DATA_PATH, index=False)
    new_test_df.to_csv(TEST_DATA_PATH, index=False)
    
    logger.info(f"Saved validation data to: {VALIDATION_DATA_PATH}")
    logger.info(f"Saved new test data to: {TEST_DATA_PATH}")
    
    return validation_df, new_test_df

