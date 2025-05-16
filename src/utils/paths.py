from pathlib import Path


PROJECT_ROOT = Path("G:/Mon Drive/ProjetTextMining")
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"


RAW_DATA_DIR = DATA_DIR / "raw_data" / "dataset1"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"
TOKENIZED_DATA_DIR = DATA_DIR / "tokenized_data"
EMPATH_DATA_DIR = DATA_DIR /"EMPATHdataset1"

TRAIN_DATA_PATH = RAW_DATA_DIR / "train_en.csv"
TEST_DATA_PATH = RAW_DATA_DIR / "test_en.csv"


VALIDATION_DATA_PATH = RAW_DATA_DIR / "validation.csv"
NEW_TEST_DATA_PATH = RAW_DATA_DIR / "new_test.csv"


MODELS_DIR = PROJECT_ROOT / "models"
GPT2_MODEL_DIR = MODELS_DIR / "gpt2"
DT_MODEL_DIR = MODELS_DIR / "decision_tree" / "dataset1"



WRITEPRINTS_DATA_DIR = DATA_DIR / "writeprints_features"
WRITEPRINTS_MODEL_DIR = MODELS_DIR / "writeprints_rfc"
LL_MODEL_DIR= MODELS_DIR / "LL"
Rank_MODEL_DIR = MODELS_DIR / "Rank"

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, TOKENIZED_DATA_DIR, 
                 GPT2_MODEL_DIR, DT_MODEL_DIR, EMPATH_DATA_DIR,WRITEPRINTS_DATA_DIR, WRITEPRINTS_MODEL_DIR,Rank_MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
