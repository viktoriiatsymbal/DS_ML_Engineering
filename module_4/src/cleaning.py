import pandas as pd
from src.config import INTERIM_DIR

def clean_data(merged_path):
    df = pd.read_csv(merged_path)
    df = df.drop_duplicates()
    df = df.dropna()

    numeric_columns = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "quality",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_columns)
    df = df[df["quality"].between(3, 8)]
    df["quality_class"] = (df["quality"] >= 6).astype(int)
    output_path = INTERIM_DIR/"clean_wine_data.csv"
    df.to_csv(output_path, index=False)
    return str(output_path)
