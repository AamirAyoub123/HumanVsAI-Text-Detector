import pandas as pd
import numpy as np
import re
import joblib
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import json
import sys
import os

# Configure NLTK
nltk.download(['stopwords', 'wordnet', 'omw-1.4'])
current_file_path = Path(__file__).resolve()
PROJECT_ROOT = current_file_path.parent.parent.parent 
sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import (
    PROCESSED_DATA_DIR,
    MODELS_DIR
)

class LemmaTFIDF_LogReg:
    def __init__(self, dataset_name="dataset1"):
        self.dataset_name = dataset_name
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                tokenizer=self._lemma_tokenizer,
                token_pattern=None,
                ngram_range=(1, 3),  # 1. Added trigrams
                max_features=15000,   # 2. Reduced feature space
                min_df=5,            # 2. Filter rare terms
                max_df=0.85,          # 2. Remove common terms
                sublinear_tf=True
            )),
            ('clf', LogisticRegression(
                C=0.5,               # 3. Stronger regularization
                penalty='l2',        # 3. L2 regularization
                class_weight={0: 2.0, 1: 1.0},  # 1. Class weighting
                solver='saga',
                max_iter=2000,       # 3. More iterations
                n_jobs=-1
            ))
        ])

    def _lemma_tokenizer(self, text):
        text = re.sub('[^a-zA-Z]', ' ', text)
        words = text.lower().split()
        return [self.lemmatizer.lemmatize(word) for word in words 
                if word not in self.stop_words]

    def load_data(self, split):
        """Load processed data from project structure"""
        texts_path = PROCESSED_DATA_DIR / "ProcessedDatasets1" / f"{split}_cleaned.csv"
        df = pd.read_csv(texts_path)
        return df['cleaned_text'].tolist(), df['label'].values

    def train(self):
        """Full training workflow with evaluation"""
        print("\nLoading datasets...")
        train_texts, y_train = self.load_data('train')
        val_texts, y_val = self.load_data('validation')
        test_texts, y_test = self.load_data('test')

        print("Training TF-IDF Logistic Regression model...")
        self.pipeline.fit(train_texts, y_train)

        results = {
            'train': self._evaluate(train_texts, y_train),
            'validation': self._evaluate(val_texts, y_val),
            'test': self._evaluate(test_texts, y_test),
        }
        
        self._save_model(results)
        return results

    def _evaluate(self, texts, y_true):
        """Generate evaluation metrics"""
        y_pred = self.pipeline.predict(texts)
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': pd.crosstab(y_true, y_pred).to_dict()
        }

    def _save_model(self, results):
        """Save model and metrics with versioning"""
        save_dir = MODELS_DIR / "LogisticRegressionTFIDF" /self.dataset_name 
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        joblib.dump(self.pipeline, save_dir / f"model_{timestamp}.joblib")
        
        with open(save_dir / f"metrics_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nModel artifacts saved to: {save_dir}")

def train_model(dataset_name):
    print(f"\n{'='*50}")
    print(f"TRAINING LEMMATIZED TF-IDF LOGISTIC REGRESSION")
    print(f"{'='*50}")
    
    detector = LemmaTFIDF_LogReg(dataset_name)
    results = detector.train()
    
    print("\nPerformance Summary:")
    for split in ['train', 'validation', 'test']:
        print(f"\n{split.upper()} SET:")
        print(f"Accuracy: {results[split]['accuracy']:.4f}")
        
        # Classification report
        report = pd.DataFrame(results[split]['report']).transpose()
        print(f"\nClassification Report:\n{report.round(4)}\n")
        
        # Confusion matrix
        cm = pd.DataFrame(results[split]['confusion_matrix'])
        print(f"Confusion Matrix:\n{cm}\n")
    
    return results

if __name__ == "__main__":
    train_model("dataset1")