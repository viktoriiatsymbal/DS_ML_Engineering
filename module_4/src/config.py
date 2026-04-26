from pathlib import Path

DATA_DIR = Path("/opt/airflow/data")
RAW_DIR = DATA_DIR/"raw"
INTERIM_DIR = DATA_DIR/"interim"
PROCESSED_DIR = DATA_DIR/"processed"
FOLDS_DIR = DATA_DIR/"folds"
REPORTS_DIR = DATA_DIR/"reports"

for d in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FOLDS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
