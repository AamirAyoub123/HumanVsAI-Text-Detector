import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import json
import sys
import torch
import os
from tqdm import tqdm  
from memory_profiler import profile  


torch.set_num_threads(os.cpu_count() or 4)
torch.set_flush_denormal(True)
print(f"PyTorch using {torch.get_num_threads()} CPU threads")

current_file_path = Path(__file__).resolve()
PROJECT_ROOT = current_file_path.parent.parent.parent 
sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import (
    RAW_DATA_DIR, 
    TRAIN_DATA_PATH,
    LL_MODEL_DIR,
    VALIDATION_DATA_PATH,
    NEW_TEST_DATA_PATH,
    PROCESSED_DATA_DIR,
    EMPATH_DATA_DIR
)

class LogLikelihoodDetector:
    def __init__(self, dataset_name="dataset1"):
        self.dataset_name = dataset_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = None
        self.lm_model = None
        self.classifier = LogisticRegression(max_iter=1000, n_jobs=-1)  
        self._load_language_model()
        
    def _load_language_model(self):
        """Load GPT-2 model with padding token configured"""
        print("Loading GPT-2 model...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        
        # Set padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  
        
        # Load with optimized settings
        self.lm_model = GPT2LMHeadModel.from_pretrained(
            'gpt2-medium',
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        # Apply model optimizations
        self.lm_model.eval()
        if self.device == 'cpu':
            self.lm_model = torch.quantization.quantize_dynamic(
                self.lm_model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            
    def _calculate_log_probs(self, text):
        """Calculate log probabilities with memory optimization"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=384).to(self.device)
        
        with torch.no_grad():
            if self.device == 'cpu':
                with torch.jit.optimized_execution(True):
                    outputs = self.lm_model(**inputs)
            else:
                outputs = self.lm_model(**inputs)
                
        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        token_log_probs = log_probs.gather(2, inputs['input_ids'].unsqueeze(-1)).squeeze(-1)
        return token_log_probs.mean().item()
    
    def load_data(self, split):
        """Load and preprocess data efficiently"""
        texts_path = PROCESSED_DATA_DIR / "ProcessedDatasets1" / f"{split}_cleaned.csv"
        df = pd.read_csv(texts_path)
        return df['cleaned_text'].tolist(), df['label'].values
    
    def extract_features(self, texts, batch_size=4):
        features = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting features"):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                padding='max_length',  
                truncation=True,
                max_length=384,
                add_special_tokens=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.lm_model(**inputs)
                log_probs = torch.log_softmax(outputs.logits, dim=-1)
                token_probs = log_probs.gather(2, inputs['input_ids'].unsqueeze(-1)).squeeze(-1)
                batch_features = token_probs.mean(dim=1).cpu().numpy()
                
            features.extend(batch_features)
            
        return np.array(features).reshape(-1, 1)
    
    
    def train(self):
        """Optimized training process"""
        print("\nLoading datasets...")
        train_texts, y_train = self.load_data('train')
        val_texts, y_val = self.load_data('validation')
        test_texts, y_test = self.load_data('test')
        
        print("Extracting features...")
        X_train = self.extract_features(train_texts)
        X_val = self.extract_features(val_texts)
        X_test = self.extract_features(test_texts)
        
        print("Training classifier...")
        self.classifier.fit(X_train, y_train)
        
        results = {
            'train': self._evaluate(X_train, y_train),
            'validation': self._evaluate(X_val, y_val),
            'test': self._evaluate(X_test, y_test),
        }
        
        self._save_model_and_artifacts(results)
        return results
    
    def _evaluate(self, X, y):
        """Compact evaluation metrics"""
        y_pred = self.classifier.predict(X)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'report': classification_report(y, y_pred, output_dict=True),
            'confusion_matrix': pd.crosstab(y, y_pred).to_dict()
        }
    
    def _save_model_and_artifacts(self, results):
        """Enhanced saving with versioning"""
        save_dir = LL_MODEL_DIR / self.dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with version timestamp
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_path = save_dir / f"model_{timestamp}.joblib"
        
        joblib.dump({
            'classifier': self.classifier,
            'config': {
                'device': self.device,
                'dataset': self.dataset_name,
                'timestamp': timestamp
            }
        }, model_path)
        
        with open(save_dir / f"results_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nSaved model and artifacts to: {save_dir}")
        print(f"- Model: {model_path}")
        print(f"- Results: results_{timestamp}.json")

def train_new_model(dataset_name):
    print(f"\n{'='*50}")
    print(f"TRAINING LOG LIKELIHOOD DETECTOR FOR {dataset_name.upper()}")
    print(f"{'='*50}")
    
    detector = LogLikelihoodDetector(dataset_name)
    results = detector.train()
    
    # Enhanced results display
    print("\nFinal Results Summary:")
    metrics = ['accuracy', 'precision', 'recall', 'f1-score']
    
    for split in ['train', 'validation', 'test']:
        report = results[split]['report']
        print(f"\n{split.upper():<12} Accuracy: {results[split]['accuracy']:.4f}")
        print(f"{'Class 0':<12} Precision: {report['0']['precision']:.4f} | Recall: {report['0']['recall']:.4f} | F1: {report['0']['f1-score']:.4f}")
        print(f"{'Class 1':<12} Precision: {report['1']['precision']:.4f} | Recall: {report['1']['recall']:.4f} | F1: {report['1']['f1-score']:.4f}")
    
    return results

if __name__ == "__main__":
    
    if os.name == 'nt':
        os.system('start /B /WAIT /HIGH python -c "import os; os.system(\"powershell -command \\\"$Process = Get-Process -Id $PID; $Process.ProcessorAffinity=15\\\"\")"')
    
    train_new_model("dataset1")