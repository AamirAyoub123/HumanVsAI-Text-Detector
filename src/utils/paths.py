from pathlib import Path

# Base paths (adjust these according to your actual VS Code workspace root)
PROJECT_ROOT = Path("G:/Mon Drive/ProjetTextMining")
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw_data" / "dataset1"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
TOKENIZED_DATA_DIR = DATA_DIR / "tokenized_data"
EMPATH_DATA_DIR = DATA_DIR /"EMPATHdataset1"
# File paths
TRAIN_DATA_PATH = RAW_DATA_DIR / "train_en.csv"
TEST_DATA_PATH = RAW_DATA_DIR / "test_en.csv"

# In src/data_loading/paths.py
VALIDATION_DATA_PATH = RAW_DATA_DIR / "validation.csv"
NEW_TEST_DATA_PATH = RAW_DATA_DIR / "new_test.csv"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
GPT2_MODEL_DIR = MODELS_DIR / "gpt2"
DT_MODEL_DIR = MODELS_DIR / "decision_tree" / "dataset1"


# Add to your paths.py
WRITEPRINTS_DATA_DIR = DATA_DIR / "writeprints_features"
WRITEPRINTS_MODEL_DIR = MODELS_DIR / "writeprints_rfc"
LL_MODEL_DIR= MODELS_DIR / "LL"
Rank_MODEL_DIR = MODELS_DIR / "Rank"

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, TOKENIZED_DATA_DIR, 
                 GPT2_MODEL_DIR, DT_MODEL_DIR, EMPATH_DATA_DIR,WRITEPRINTS_DATA_DIR, WRITEPRINTS_MODEL_DIR,Rank_MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
# Example usage in other files:
# from src.data_loading.paths import TRAIN_DATA_PATH, GPT2_MODEL_DIR