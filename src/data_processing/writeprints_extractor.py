import pandas as pd
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm
import sys
import os
current_file_path = Path(__file__).resolve()
PROJECT_ROOT = current_file_path.parent.parent.parent 
sys.path.append(str(PROJECT_ROOT))
from src.utils.paths import (
    WRITEPRINTS_DATA_DIR,
    PROCESSED_DATA_DIR
)

class WriteprintsExtractor:
    def __init__(self):
        self.features = [
            'word_count', 'char_count', 'avg_word_length',
            'type_token_ratio', 'digit_count', 'uppercase_ratio',
            'punctuation_count', 'special_char_count',
            'function_word_ratio', 'sentence_length_var'
        ]
    
    def extract_features(self, text):
        """Extract writeprints features from a single text"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {f: 0 for f in self.features}
        
        # Basic counts
        words = re.findall(r'\w+', text)
        chars = list(text)
        sentences = re.split(r'[.!?]+', text)
        
        # Feature calculations
        features = {
            'word_count': len(words),
            'char_count': len(chars),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'type_token_ratio': len(set(words))/len(words) if words else 0,
            'digit_count': sum(c.isdigit() for c in chars),
            'uppercase_ratio': sum(c.isupper() for c in chars)/len(chars) if chars else 0,
            'punctuation_count': sum(c in '.,;:!?-' for c in chars),
            'special_char_count': sum(not c.isalnum() for c in chars),
            'function_word_ratio': self._calc_function_word_ratio(words),
            'sentence_length_var': np.var([len(s.split()) for s in sentences if s]) if sentences else 0
        }
        return features
    
    def _calc_function_word_ratio(self, words):
        function_words = {'the', 'and', 'of', 'to', 'in', 'is', 'it', 'that', 'for', 'with'}
        return sum(w.lower() in function_words for w in words)/len(words) if words else 0
    
    def process_dataset(self, dataset_name):
        splits = ['train', 'validation', 'test']
        
        for split in splits:
            # Load processed texts
            data_path = PROCESSED_DATA_DIR / "ProcessedDatasets1" / f"{split}_cleaned.csv"
            df = pd.read_csv(data_path)
            
            # Extract features
            features = []
            for text in tqdm(df['cleaned_text'], desc=f"Processing {split}"):
                features.append(self.extract_features(text))
            
            # Save features
            feature_df = pd.DataFrame(features)
            output_path = WRITEPRINTS_DATA_DIR / "dataset1" /f"writeprints_{split}.csv"
            feature_df.to_csv(output_path, index=False)
            print(f"Saved {split} features to {output_path}")