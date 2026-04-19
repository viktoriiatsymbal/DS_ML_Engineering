import os
import json
from joblib import load
from sklearn.datasets import load_wine

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/best_model.joblib")
FEATURES_PATH = os.getenv("FEATURES_PATH", "artifacts/best_features.json")

def main():
    wine = load_wine(as_frame=True)
    X = wine.data.copy()

    bundle = load(MODEL_PATH)
    model = bundle["model"]
    feature_columns = bundle["features"]

    X_subset = X.loc[:, feature_columns].copy()
    X_subset = X_subset[feature_columns]
    X_subset = X_subset.head(5)

    preds = model.predict(X_subset)
    out = X_subset.copy()
    out["prediction"] = preds
    print(out)

if __name__ == "__main__":
    main()
