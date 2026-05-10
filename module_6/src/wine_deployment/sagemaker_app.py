import sys
import os
import time
import mlflow
import mlflow.pyfunc
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
model = None

def load_model():
    global model

    if model is not None:
        return model

    model_dir = os.getenv("MODEL_DIR", "/opt/ml/model")
    model_uri = os.getenv("MODEL_URI")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if model_uri:
        model = mlflow.pyfunc.load_model(model_uri)
    else:
        model = mlflow.pyfunc.load_model(model_dir)
    return model

def payload_to_dataframe(payload):
    if isinstance(payload, dict) and "dataframe_split" in payload:
        return pd.DataFrame(
            data=payload["dataframe_split"]["data"],
            columns=payload["dataframe_split"]["columns"])

    if isinstance(payload, dict) and "instances" in payload:
        return pd.DataFrame(payload["instances"])

    if isinstance(payload, dict) and "columns" in payload and "data" in payload:
        return pd.DataFrame(data=payload["data"], columns=payload["columns"])

    if isinstance(payload, dict):
        return pd.DataFrame([payload])

    if isinstance(payload, list):
        return pd.DataFrame(payload)

    raise ValueError(
        "Input must be a JSON object, list of objects, "
        "{'instances': [...]}, {'columns': [...], 'data': [...]}, "
        "or {'dataframe_split': {'columns': [...], 'data': [...]}}.")

@app.route("/ping", methods=["GET"])
def ping():
    try:
        load_model()
        return "", 200
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500

@app.route("/invocations", methods=["POST"])
def invocations():
    started = time.perf_counter()

    try:
        current_model = load_model()
        payload = request.get_json(force=True)
        input_df = payload_to_dataframe(payload)

        predictions = current_model.predict(input_df)
        latency_ms = round((time.perf_counter() - started) * 1000, 3)

        return jsonify(
            {
                "success": True,
                "predictions": [int(pred) for pred in predictions],
                "n_rows": int(len(input_df)),
                "latency_ms": latency_ms,
            }
        )

    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400

def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if command != "serve":
        raise ValueError(
            f"Unsupported command: {command}. "
            "This container only supports the 'serve' command")

    print("Loading model and starting SageMaker-compatible Flask server...")
    load_model()
    app.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", "8080")))

if __name__ == "__main__":
    main()
