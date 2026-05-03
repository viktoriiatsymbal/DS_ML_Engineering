import os
import time
from typing import Any
import mlflow.pyfunc
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
model = None

def load_model():
    global model
    model_dir = os.getenv("MODEL_DIR")
    model_uri = os.getenv("MODEL_URI")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if model_dir:
        model = mlflow.pyfunc.load_model(model_dir)
        return

    if model_uri:
        model = mlflow.pyfunc.load_model(model_uri)
        return

    raise RuntimeError("MODEL_DIR or MODEL_URI must be provided.")


def payload_to_dataframe(payload):
    if isinstance(payload, dict) and "instances" in payload:
        return pd.DataFrame(payload["instances"])

    if isinstance(payload, dict):
        return pd.DataFrame([payload])

    if isinstance(payload, list):
        return pd.DataFrame(payload)

    raise ValueError("Input must be a JSON object, list of objects, or {'instances': [...]}.")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    started = time.perf_counter()
    try:
        payload = request.get_json(force=True)
        input_df = payload_to_dataframe(payload)
        predictions = model.predict(input_df)
        latency_ms = round((time.perf_counter() - started) * 1000, 3)

        return jsonify(
            {
                "success": True,
                "predictions": [int(pred) for pred in predictions],
                "n_rows": int(len(input_df)),
                "latency_ms": latency_ms,
            })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

if __name__ == "__main__":
    print("Loading model and starting Flask server...")
    load_model()
    app.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", "8000")))
