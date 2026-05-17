from pathlib import Path
import pandas as pd
import yaml

def load_params(path="params.yaml"):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def main():
    params = load_params()
    raw_path = Path(params["data"]["raw_path"])
    output_path = Path(params["preprocess_basic"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path, sep=";")
    df = df.drop_duplicates()

    numeric_columns = [
        "fixed acidity", "volatile acidity", "citric acid",
        "residual sugar", "chlorides", "free sulfur dioxide",
        "total sulfur dioxide", "density", "pH",
        "sulphates", "alcohol", "quality"]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_columns)
    df = df[df["quality"].between(3, 8)]
    df["quality_class"] = (df["quality"] >= 6).astype(int)
    df["alcohol_segment"] = pd.cut(
        df["alcohol"],
        bins=[0, 10, 12, 100],
        labels=["low_alcohol", "medium_alcohol", "high_alcohol"]).astype(str)
    df.to_csv(output_path, index=False)
    print(f"Saved basic preprocessed data to {output_path}")

if __name__ == "__main__":
    main()
