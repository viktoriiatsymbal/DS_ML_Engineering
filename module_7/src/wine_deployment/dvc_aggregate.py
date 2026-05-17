from pathlib import Path
import pandas as pd
import yaml

def load_params(path="params.yaml"):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def main():
    params = load_params()
    input_path = Path(params["preprocess_aggregated"]["input_path"])
    output_path = Path(params["preprocess_aggregated"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)

    aggregated = (
        df.groupby(["quality", "alcohol_segment"], dropna=False)
        .agg(
            rows_count=("quality", "count"),
            avg_alcohol=("alcohol", "mean"),
            avg_sulphates=("sulphates", "mean"),
            avg_volatile_acidity=("volatile acidity", "mean"),
            avg_citric_acid=("citric acid", "mean"),
            avg_density=("density", "mean"),
            avg_ph=("pH", "mean")).reset_index())
    aggregated.to_csv(output_path, index=False)

    print(f"Saved aggregated data to {output_path}")

if __name__ == "__main__":
    main()
