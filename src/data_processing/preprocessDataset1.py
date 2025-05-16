import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
import sys
import os
current_file_path = Path(__file__).resolve()
PROJECT_ROOT = current_file_path.parent.parent.parent 
sys.path.append(str(PROJECT_ROOT))
from src.utils.paths import (
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR
)

def clean_text(text):
    """Enhanced text cleaning pipeline"""
    if pd.isna(text):
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # 1. Remove URLs and social media handles
    text = re.sub(r'https?://\S+|www\.\S+|@\w+|#\w+', '', text)
    
    # 2. Remove special characters (keep basic punctuation and letters)
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    
    # 3. Convert to lowercase
    text = text.lower()
    
    # 4. Remove extra whitespace and newlines
    return ' '.join(text.split())

def preprocess_dataset():
    """Process all dataset splits and save cleaned versions"""
    # Define file paths using your existing path structure
    file_map = {
        'train': RAW_DATA_DIR / 'train_en.csv',
        'validation': RAW_DATA_DIR / 'validation.csv',
        'test': RAW_DATA_DIR / 'new_test.csv'
    }
    
    for split_name, file_path in file_map.items():
        print(f"Processing {split_name} data...")
        
        # Load raw data
        df = pd.read_csv(file_path)
        
        
        cols_to_keep = ['text', 'label']
        if 'prompt' in df.columns:
            cols_to_keep.append('prompt')
        df = df[cols_to_keep]
        
        # 2. Handle missing values
        df['text'] = df['text'].fillna('')
        if 'prompt' in df.columns:
            df['prompt'] = df['prompt'].fillna('NO-PROMPT')
        
        # 3. Combine prompt + text where applicable
        if 'prompt' in df.columns:
            df['full_text'] = df.apply(
                lambda x: f"{x['prompt']} {x['text']}".strip() 
                if x['prompt'] != "NO-PROMPT" 
                else x['text'],
                axis=1
            )
        else:
            df['full_text'] = df['text']
        
        # 4. Clean text with progress bar
        tqdm.pandas(desc=f"Cleaning {split_name} texts")
        df['cleaned_text'] = df['full_text'].progress_apply(clean_text)
        
        # 5. Save processed data
        output_path = PROCESSED_DATA_DIR / "ProcessedDatasets1"/ f"{split_name}_cleaned.csv"
        df[['cleaned_text', 'label']].to_csv(output_path, index=False)
        
        print(f"Saved cleaned {split_name} data to: {output_path}")

if __name__ == "__main__":
    preprocess_dataset()
def verify_output():
    splits = ['train', 'validation', 'test']
    for split in splits:
        path = PROCESSED_DATA_DIR / "ProcessedDatasets1"/f"{split}_cleaned.csv"
        df = pd.read_csv(path)
        print(f"\n{split.upper()} SET VERIFICATION:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Sample Text: {df['cleaned_text'].iloc[0][:100]}...")
        print(f"Label Distribution:\n{df['label'].value_counts(normalize=True)}")

verify_output()