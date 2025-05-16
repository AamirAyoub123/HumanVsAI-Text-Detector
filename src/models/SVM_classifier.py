import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import json
import sys


current_file_path = Path(__file__).resolve()
PROJECT_ROOT = current_file_path.parent.parent.parent 
sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import (
    PROCESSED_DATA_DIR,
    MODELS_DIR
)

class Char3GramSVMDetector:
    def __init__(self, dataset_name="dataset1"):
        self.dataset_name = dataset_name
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4), 
            sublinear_tf=True
        )
        self.classifier = SVC(
            kernel='linear',
            C=0.7,  
            class_weight={0: 1.5, 1: 1},  
            probability=True,
            random_state=42
        )
        self.pipeline = make_pipeline(self.vectorizer, self.classifier)

    def load_data(self, split):
        """Load processed data """
        texts_path = PROCESSED_DATA_DIR / "ProcessedDatasets1" / f"{split}_cleaned.csv"
        df = pd.read_csv(texts_path)
        return df['cleaned_text'].tolist(), df['label'].values

    def train(self):
        """Full training workflow"""
        print("\nLoading datasets...")
        train_texts, y_train = self.load_data('train')
        val_texts, y_val = self.load_data('validation')
        test_texts, y_test = self.load_data('test')

        print("Training 3-gram SVM classifier...")
        self.pipeline.fit(train_texts, y_train)

        results = {
            'train': self._evaluate(train_texts, y_train),
            'validation': self._evaluate(val_texts, y_val),
            'test': self._evaluate(test_texts, y_test),
        }
        
        self._save_model_and_artifacts(results)
        return results

    def _evaluate(self, texts, y_true):
        """Evaluation method"""
        y_pred = self.pipeline.predict(texts)
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': pd.crosstab(y_true, y_pred).to_dict()
        }

    def _save_model_and_artifacts(self, results):
        """Save model with versioning"""
        save_dir = MODELS_DIR /"SVM_classifier"/self.dataset_name 
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_path = save_dir / f"model_{timestamp}.joblib"
        
        joblib.dump(self.pipeline, model_path)
        
        with open(save_dir / f"results_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2)

def train_char3gram_svm(dataset_name):
    print(f"\n{'='*50}")
    print(f"TRAINING CHARACTER 3-GRAM SVM FOR {dataset_name.upper()}")
    print(f"{'='*50}")
    
    detector = Char3GramSVMDetector(dataset_name)
    results = detector.train()
    
    print("\nResults Summary:")
    for split in ['train', 'validation', 'test']:
        report = results[split]['report']
        print(f"\n{split.upper():<12} Accuracy: {results[split]['accuracy']:.4f}")
        print(f"{'Class 0':<12} Precision: {report['0']['precision']:.4f} | Recall: {report['0']['recall']:.4f} | F1: {report['0']['f1-score']:.4f}")
        print(f"{'Class 1':<12} Precision: {report['1']['precision']:.4f} | Recall: {report['1']['recall']:.4f} | F1: {report['1']['f1-score']:.4f}")
    
    return results

if __name__ == "__main__":
    train_char3gram_svm("dataset1")