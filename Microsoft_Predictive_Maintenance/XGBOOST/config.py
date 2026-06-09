from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

RAW_DATA_DIR = ROOT_DIR / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "XGBOOST" / "data"
LOGS_DIR = ROOT_DIR / "XGBOOST" / "logs"
MODELS_DIR = ROOT_DIR / "XGBOOST" / "models"