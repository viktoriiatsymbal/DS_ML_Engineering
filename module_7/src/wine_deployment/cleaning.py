import pandas as pd
from wine_deployment.config import INTERIM_DIR

NUMERIC_COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH",
    "sulphates", "alcohol", "quality"]

def clean_data(merged_path):
    df = pd.read_csv(merged_path)
    df = df.drop_duplicates()

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=NUMERIC_COLUMNS + ["alcohol_segment"])
    df = df[df["quality"].between(3, 8)]
    df["quality_class"] = (df["quality"] >= 6).astype(int)
    output_path = INTERIM_DIR / "clean_wine_data.csv"
    df.to_csv(output_path, index=False)
    return str(output_path)
