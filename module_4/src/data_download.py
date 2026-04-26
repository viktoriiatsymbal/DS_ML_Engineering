import requests
from src.config import RAW_DIR

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv")

def download_wine_data():
    output_path = RAW_DIR / "winequality-red.csv"
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return str(output_path)
