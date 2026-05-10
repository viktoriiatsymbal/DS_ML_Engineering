import requests
from wine_deployment.config import RAW_DIR

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)

def download_wine_data():
    output_path = RAW_DIR / "winequality-red.csv"
    if output_path.exists() and output_path.stat().st_size > 0:
        return str(output_path)
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return str(output_path)
