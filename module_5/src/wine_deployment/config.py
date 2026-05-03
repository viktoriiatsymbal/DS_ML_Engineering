from pathlib import Path
import os

DATA_DIR = Path(os.getenv("APP_DATA_DIR", "/opt/airflow/data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
FOLDS_DIR = DATA_DIR / "folds"
REPORTS_DIR = DATA_DIR / "reports"

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
BUILD_DIR = PROJECT_ROOT / "build"
ONLINE_MODEL_DIR = BUILD_DIR / "online_model"

for directory in [
    RAW_DIR, INTERIM_DIR,
    PROCESSED_DIR, FOLDS_DIR,
    REPORTS_DIR, BUILD_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "module5-wine-deployment"
REGISTERED_MODEL_NAME = "wine_best_model"
TARGET_COL = "quality_class"
ORIGINAL_TARGET_COL = "quality"
DROP_COLS = ["quality", "quality_class", "row_id"]
CATEGORICAL_COLS = ["alcohol_segment"]
