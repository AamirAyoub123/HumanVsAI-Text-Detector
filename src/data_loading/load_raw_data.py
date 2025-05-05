from datasets import load_dataset
from pathlib import Path
import pandas as pd
import sys
import os
current_file_path = Path(__file__).resolve()
PROJECT_ROOT = current_file_path.parent.parent.parent 
sys.path.append(str(PROJECT_ROOT))
from src.utils.paths import (
    RAW_DATA_DIR, 
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    VALIDATION_DATA_PATH,
    NEW_TEST_DATA_PATH
)
from src.data_loading.split_data import split_test_data  # Import the splitting function

def load_and_save_data():
    """Load dataset from HuggingFace, save locally, and split test data"""
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("symanto/autextification2023", "detection_en", trust_remote_code=True)
    
    # Save raw data using the predefined paths
    dataset["train"].to_csv(TRAIN_DATA_PATH, index=False)
    dataset["test"].to_csv(TEST_DATA_PATH, index=False)
    print(f"Raw data saved to {RAW_DATA_DIR}")
    
    # Load the test data we just saved
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    # Split test data into validation and new test sets
    validation_df, new_test_df = split_test_data(test_df, 0.5)
    
    # Save the split datasets
    validation_df.to_csv(VALIDATION_DATA_PATH, index=False)
    new_test_df.to_csv(NEW_TEST_DATA_PATH, index=False)
    
    print("\nData splits saved:")
    print(f"- Train data: {TRAIN_DATA_PATH}")
    print(f"- Validation data: {VALIDATION_DATA_PATH}")
    print(f"- New test data: {NEW_TEST_DATA_PATH}")

if __name__ == "__main__":
    load_and_save_data()