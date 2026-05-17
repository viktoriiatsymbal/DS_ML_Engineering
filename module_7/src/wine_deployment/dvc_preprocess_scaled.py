from pathlib import Path
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

def load_params(path="params.yaml"):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def main():
    params = load_params()
    input_path = Path(params["preprocess_scaled"]["input_path"])
    output_path = Path(params["preprocess_scaled"]["output_path"])
    target_col = params["data"]["target_col"]
    original_target_col = params["data"]["original_target_col"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    for col in [target_col, original_target_col]:
        if col in numeric_cols:
            numeric_cols.remove(col)
    df_scaled = df.copy()

    scaler = StandardScaler()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

    df_scaled.to_csv(output_path, index=False)
    print(f"Saved scaled data to {output_path}")

if __name__ == "__main__":
    main()
