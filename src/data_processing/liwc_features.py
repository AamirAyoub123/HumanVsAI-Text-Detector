from empath import Empath
import pandas as pd
from pathlib import Path
from tqdm import tqdm
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
    NEW_TEST_DATA_PATH,
    PROCESSED_DATA_DIR,
    EMPATH_DATA_DIR
)

class EmpathAnalyzer:
    def __init__(self):
        self.lexicon = Empath()
        # Add custom categories for AI/human detection
        self._add_custom_categories()
        
    def _add_custom_categories(self):
        """Add domain-specific categories for better analysis"""
        custom_categories = {
            'ai_content': ['algorithm', 'neural network', 'transformer', 'language model', 'gpt'],
            'human_content': ['I think', 'in my opinion', 'personally', 'I feel', 'experience']
        }
        for name, terms in custom_categories.items():
            self.lexicon.create_category(name, terms)
    
    def extract_features(self, text):
        """Extract Empath features with normalization"""
        analysis = self.lexicon.analyze(text, normalize=True)
        # Add custom normalizations
        word_count = max(1, len(text.split()))  # Prevent division by zero
        return {k: v/word_count for k, v in analysis.items()}
    
    def process_dataset(self, dataset_name):
        """Process all splits for a dataset"""
        splits = ['train', 'validation', 'test']
        
        for split in splits:
            # Load cleaned data
            data_path = PROCESSED_DATA_DIR / "ProcessedDatasets1" / f"{split}_cleaned.csv"
            df = pd.read_csv(data_path)
            
            # Extract features with progress bar
            features = []
            for text in tqdm(df['cleaned_text'], desc=f"Empath {split}"):
                features.append(self.extract_features(text))
            
            # Save features
            feature_df = pd.DataFrame(features)
            output_path = EMPATH_DATA_DIR / dataset_name / f"empath_{split}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            feature_df.to_csv(output_path, index=False)
            print(f"Saved Empath features to {output_path}")

analyzer = EmpathAnalyzer()

# Process dataset1
analyzer.process_dataset("dataset1")

# Verify output
train_features = pd.read_csv(EMPATH_DATA_DIR / "dataset1" / "empath_train.csv")
print(train_features.head())