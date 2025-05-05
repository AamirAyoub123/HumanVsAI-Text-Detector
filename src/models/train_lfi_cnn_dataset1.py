import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import json 
import sys
from pathlib import Path
import os
current_file_path = Path(__file__).resolve()
PROJECT_ROOT = current_file_path.parent.parent.parent 
sys.path.append(str(PROJECT_ROOT))
from src.utils.paths import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    WRITEPRINTS_MODEL_DIR
)
from src.data_processing.lfi_feature_extractordataset1 import LFIExtractor



class LFICNN:
    def __init__(self, input_shape=(21, 1)):
        self.model = self._build_model(input_shape)
    
    def _build_model(self, input_shape):
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape),
            MaxPooling1D(2),
            Conv1D(128, 3, activation='relu'),
            MaxPooling1D(2),
            Conv1D(256, 3, activation='relu'),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                   tf.keras.metrics.AUC(name='auc'),
                   tf.keras.metrics.Precision(name='precision'),
                   tf.keras.metrics.Recall(name='recall')]
        )
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
            ]
        )
        # Save model in dataset1 subfolder
        model_path = MODELS_DIR  / "lfi_cnn.h5"/ "dataset1"/"lfi_cnn.h5"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(model_path)
        return history
    
    def evaluate(self, X_test, y_test):
        # Get predictions
        y_pred = (self.model.predict(X_test) > 0.5).astype("int32")
        
        # Calculate metrics
        test_loss, test_acc, test_auc, test_precision, test_recall = self.model.evaluate(X_test, y_test, verbose=0)
        test_f1 = f1_score(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'loss': test_loss,
            'accuracy': test_acc,
            'auc': test_auc,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'classification_report': report
        }

def prepare_data():
    extractor = LFIExtractor()
    
    # Process all splits
    splits = {}
    for split in ['train', 'validation', 'test']:
        features = extractor.process_dataset(split)
        labels = pd.read_csv(PROCESSED_DATA_DIR / "ProcessedDatasets1" / f"{split}_cleaned.csv")['label']
        splits[split] = (features, labels)
    
    # Reshape for CNN
    X_train = np.expand_dims(splits['train'][0].values, axis=-1)
    y_train = splits['train'][1].values
    X_val = np.expand_dims(splits['validation'][0].values, axis=-1)
    y_val = splits['validation'][1].values
    X_test = np.expand_dims(splits['test'][0].values, axis=-1)
    y_test = splits['test'][1].values
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def save_results(results, history):
    """Save metrics and training history to JSON files"""
    # Create results directory if not exists
    results_dir = MODELS_DIR  / "lfi_cnn"/ "dataset1"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save evaluation metrics
    with open(results_dir / "lfi_cnn_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save training history
    with open(results_dir / "lfi_cnn_history.json", 'w') as f:
        json.dump(history.history, f, indent=2)

def train_and_evaluate():
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    
    model = LFICNN(input_shape=(X_train.shape[1], 1))
    history = model.train(X_train, y_train, X_val, y_val)
    
    print("\nTest Evaluation:")
    results = model.evaluate(X_test, y_test)
    
    # Print and save results
    print(f"Test Loss: {results['loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test AUC: {results['auc']:.4f}")
    print(f"Test F1 Score: {results['f1_score']:.4f}")
    print("\nClassification Report:")
    print(pd.DataFrame(results['classification_report']).transpose())
    
    save_results(results, history)

if __name__ == "__main__":
    train_and_evaluate()