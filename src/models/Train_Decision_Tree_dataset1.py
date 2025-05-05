import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path
import json
import sys
import os

current_file_path = Path(__file__).resolve()
PROJECT_ROOT = current_file_path.parent.parent.parent 
sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import (
    RAW_DATA_DIR, 
    TRAIN_DATA_PATH,
    DT_MODEL_DIR,  # This should point to G:\...\src\models\decision_tree\dataset1\
    VALIDATION_DATA_PATH,
    NEW_TEST_DATA_PATH,
    PROCESSED_DATA_DIR,
    EMPATH_DATA_DIR
)

class DecisionTreeTrainer:
    def __init__(self, dataset_name="dataset1"):
        self.dataset_name = dataset_name
        self.model = DecisionTreeClassifier(
            max_depth=None,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        )
        
    def load_data(self, split):
        """Load features and labels for a dataset split"""
        features_path = EMPATH_DATA_DIR / self.dataset_name / f"empath_{split}.csv"
        labels_path = PROCESSED_DATA_DIR / "ProcessedDatasets1" / f"{split}_cleaned.csv"
        
        X = pd.read_csv(features_path)
        y = pd.read_csv(labels_path)['label']
        return X, y
    
    def train(self):
        """Train model and evaluate on all splits"""
        # Load all data
        X_train, y_train = self.load_data('train')
        X_val, y_val = self.load_data('validation')
        X_test, y_test = self.load_data('test')
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        results = {
            'train': self._evaluate(X_train, y_train),
            'validation': self._evaluate(X_val, y_val),
            'test': self._evaluate(X_test, y_test),
            'features': list(X_train.columns),
            'feature_importances': dict(zip(
                X_train.columns,
                self.model.feature_importances_
            ))
        }
        
        # Save everything
        self._save_model_and_artifacts(results)
        return results
    
    def _evaluate(self, X, y):
        y_pred = self.model.predict(X)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'report': classification_report(y, y_pred, output_dict=True),
            'confusion_matrix': pd.crosstab(y, y_pred).to_dict()
        }
    
    def _save_model_and_artifacts(self, results):
        """Save all training artifacts directly to dataset1 folder"""
        # Ensure directory exists
        save_dir = DT_MODEL_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save trained model
        joblib.dump(self.model, save_dir / "model.joblib")
        
        # 2. Save evaluation results
        with open(save_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # 3. Save text representation of tree
        with open(save_dir / "tree_rules.txt", 'w') as f:
            f.write(export_text(self.model, feature_names=results['features']))
        
        print(f"Saved all artifacts to: {save_dir}")
        print("Files created:")
        print(f"- {save_dir/'model.joblib'}")
        print(f"- {save_dir/'results.json'}")
        print(f"- {save_dir/'tree_rules.txt'}")

def train_new_model(dataset_name):
    print(f"\n{'='*50}")
    print(f"TRAINING DECISION TREE FOR {dataset_name.upper()}")
    print(f"{'='*50}")
    
    trainer = DecisionTreeTrainer(dataset_name)
    results = trainer.train()
    
    # Print summary
    print("\nTraining Results:")
    for split in ['train', 'validation', 'test']:
        print(f"\n{split.upper()} SET:")
        print(f"Accuracy: {results[split]['accuracy']:.4f}")

    for split in ['train', 'validation', 'test']:
        print(f"\n{split.upper()} SET:")
        report = results[split]['report']
        print(f"Accuracy: {results[split]['accuracy']:.4f}")
        print(f"Class 0 F1-score: {report['0']['f1-score']:.4f}")
        print(f"Class 1 F1-score: {report['1']['f1-score']:.4f}")
        print(f"Macro-F1: {report['macro avg']['f1-score']:.4f}")

    return results

if __name__ == "__main__":
    train_new_model("dataset1")
    
