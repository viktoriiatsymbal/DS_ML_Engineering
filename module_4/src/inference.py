import os
import mlflow.pyfunc
import pandas as pd
from src.config import REPORTS_DIR

def run_inference_from_registry(model_uri, transformed_paths):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    X_test = pd.read_csv(transformed_paths["X_test"]).head(10)
    model = mlflow.pyfunc.load_model(model_uri)
    predictions = model.predict(X_test)

    output = X_test.copy()
    output["prediction"] = predictions
    output_path = REPORTS_DIR/"sample_predictions.csv"
    output.to_csv(output_path, index=False)
    return str(output_path)
