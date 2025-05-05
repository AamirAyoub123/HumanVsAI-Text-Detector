from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import joblib
from pathlib import Path
import json 
import sys
import os
current_file_path = Path(__file__).resolve()
PROJECT_ROOT = current_file_path.parent.parent.parent 
sys.path.append(str(PROJECT_ROOT))
from src.utils.paths import (
    WRITEPRINTS_DATA_DIR,
    PROCESSED_DATA_DIR,
    WRITEPRINTS_MODEL_DIR
)
from src.data_processing.writeprints_extractor import WriteprintsExtractor

class WriteprintsRFC:
    def __init__(self):
        self.model = RandomForestClassifier(
           n_estimators=100,          # Increased from 100
            max_depth=8,              # Reduced from unlimited
            min_samples_split=10,      # Increased from 2
            min_samples_leaf=5,        # Increased from 1
            max_features='sqrt',       # Reduced from 'auto'
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    
    def train(self):
        # Load features and labels
        X_train = pd.read_csv(WRITEPRINTS_DATA_DIR / "dataset1"/"writeprints_train.csv")
        y_train = pd.read_csv(PROCESSED_DATA_DIR / "ProcessedDatasets1" / "train_cleaned.csv")['label']
        
        X_val = pd.read_csv(WRITEPRINTS_DATA_DIR /"dataset1" /"writeprints_validation.csv")
        y_val = pd.read_csv(PROCESSED_DATA_DIR / "ProcessedDatasets1" / "validation_cleaned.csv")['label']
        

        X_test = pd.read_csv(WRITEPRINTS_DATA_DIR /"dataset1" /"writeprints_test.csv")
        y_test = pd.read_csv(PROCESSED_DATA_DIR / "ProcessedDatasets1" / "test_cleaned.csv")['label']
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        results = {
            'train': classification_report(y_train, self.model.predict(X_train), output_dict=True),
            'validation': classification_report(y_val, self.model.predict(X_val), output_dict=True),
            'test':classification_report(y_test,self.model.predict(X_test),output_dict=True)
        }
        
        # Save model
        model_path = WRITEPRINTS_MODEL_DIR /"dataset1" /"model.joblib"
        joblib.dump(self.model, model_path)
        
        # Save results
        with open(WRITEPRINTS_MODEL_DIR /"dataset1" / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Model saved to {model_path}")
        return results

def train_writeprints():
    print("Extracting Writeprints features...")
    extractor = WriteprintsExtractor()
    extractor.process_dataset("dataset1")
    
    print("\nTraining Random Forest Classifier...")
    rfc = WriteprintsRFC()
    results = rfc.train()
    
    print("\nTraining Results:")
    print(pd.DataFrame({
        'Train Accuracy': [results['train']['accuracy']],
        'Validation Accuracy': [results['validation']['accuracy']],
        'test Accuracy': [results['test']['accuracy']]
    }))

if __name__ == "__main__":
    train_writeprints()