import os
import time
import mlflow.pyfunc
import pandas as pd
from wine_deployment.config import MLFLOW_TRACKING_URI, REPORTS_DIR

def run_batch_inference(model_uri, split_paths):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI))

    X_test = pd.read_csv(split_paths["X_test_raw"])
    model = mlflow.pyfunc.load_model(model_uri)

    started = time.perf_counter()
    predictions = model.predict(X_test)
    latency_seconds = time.perf_counter() - started

    output = X_test.copy()
    output["prediction"] = predictions
    output_path = REPORTS_DIR / "batch_predictions.csv"
    output.to_csv(output_path, index=False)

    latency_path = REPORTS_DIR / "batch_latency.txt"
    latency_path.write_text(
        f"rows={len(X_test)}\nlatency_seconds={latency_seconds:.6f}\n",
        encoding="utf-8")
    return str(output_path)
