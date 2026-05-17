from pathlib import Path
import json
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from dvclive import Live
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from wine_deployment.training import build_models, make_pipeline

def load_params(path="params.yaml"):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def configure_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("module7-dvc-wine-experiment")

def main():
    params = load_params()
    configure_mlflow()
    input_path = Path(params["train"]["input_path"])
    results_path = Path(params["train"]["results_path"])
    best_info_path = Path(params["train"]["best_info_path"])
    model_uri_path = Path(params["train"]["model_uri_path"])

    target_col = params["data"]["target_col"]
    original_target_col = params["data"]["original_target_col"]
    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]
    n_splits = params["train"]["n_splits"]

    results_path.parent.mkdir(parents=True, exist_ok=True)
    best_info_path.parent.mkdir(parents=True, exist_ok=True)
    model_uri_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)

    drop_cols = [target_col]
    if original_target_col in df.columns:
        drop_cols.append(original_target_col)

    X = df.drop(columns=drop_cols)
    y = df[target_col]
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state)

    results = []
    best_score = -1.0
    best_model_name = None
    best_pipeline = None
    best_test_f1 = None
    best_test_accuracy = None
    best_model_params = None

    for model_name, model, model_params in build_models():
        fold_f1_scores = []
        fold_accuracy_scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            pipeline = make_pipeline(clone(model), feature_columns)
            pipeline.fit(X_fold_train, y_fold_train)

            val_preds = pipeline.predict(X_fold_val)

            fold_f1_scores.append(f1_score(y_fold_val, val_preds))
            fold_accuracy_scores.append(accuracy_score(y_fold_val, val_preds))

        mean_cv_f1 = round(float(sum(fold_f1_scores) / len(fold_f1_scores)), 4)
        mean_cv_accuracy = round(
            float(sum(fold_accuracy_scores) / len(fold_accuracy_scores)), 4)

        final_pipeline = make_pipeline(clone(model), feature_columns)
        final_pipeline.fit(X_train, y_train)

        test_preds = final_pipeline.predict(X_test)
        test_f1 = round(float(f1_score(y_test, test_preds)), 4)
        test_accuracy = round(float(accuracy_score(y_test, test_preds)), 4)

        result = {
            "model_name": model_name,
            "mean_cv_f1": mean_cv_f1,
            "mean_cv_accuracy": mean_cv_accuracy,
            "test_f1": test_f1,
            "test_accuracy": test_accuracy,
            **model_params,
        }

        results.append(result)

        if mean_cv_f1 > best_score:
            best_score = mean_cv_f1
            best_model_name = model_name
            best_pipeline = final_pipeline
            best_test_f1 = test_f1
            best_test_accuracy = test_accuracy
            best_model_params = model_params

    results_df = pd.DataFrame(results).sort_values("mean_cv_f1", ascending=False)
    results_df.to_csv(results_path, index=False)

    input_example = X_test.head(3)
    signature = infer_signature(input_example, best_pipeline.predict(input_example))

    with mlflow.start_run(run_name=f"dvc_{best_model_name}") as run:
        mlflow.set_tag("project", "module7-dvc-data-governance")
        mlflow.set_tag("artifact_type", "full_pipeline_preprocessor_plus_model")

        if best_model_params:
            for key, value in best_model_params.items():
                mlflow.log_param(key, value)

        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        mlflow.log_metric("best_cv_f1", float(best_score))
        mlflow.log_metric("best_test_f1", float(best_test_f1))
        mlflow.log_metric("best_test_accuracy", float(best_test_accuracy))

        model_info = mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name="model",
            signature=signature,
            input_example=input_example,
        )

        run_id = run.info.run_id
        model_uri = model_info.model_uri

    metrics = {
        "best_model_name": best_model_name,
        "best_cv_f1": float(best_score),
        "best_test_f1": float(best_test_f1),
        "best_test_accuracy": float(best_test_accuracy),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "n_models_tested": int(len(results)),
    }

    with Live("dvclive", dvcyaml=False) as live:
        for metric_name, metric_value in metrics.items():
            live.log_metric(metric_name, metric_value)

    best_info = {
        "best_model_name": best_model_name,
        "best_model_params": best_model_params,
        "run_id": run_id,
        "model_uri": model_uri,
        "results_path": str(results_path),
        "metrics": metrics,
    }

    with open(best_info_path, "w", encoding="utf-8") as file:
        json.dump(best_info, file, indent=2)
    
    model_uri_path.write_text(str(model_uri) + "\n", encoding="utf-8")

    print(f"Best model: {best_model_name}")
    print(f"MLflow model URI: {model_uri}")
    print(f"Saved model comparison to: {results_path}")
    print(f"Saved best model info to: {best_info_path}")
    print(f"Saved model URI to: {model_uri_path}")
    print("Saved DVC metrics to: dvclive/metrics.json")

if __name__ == "__main__":
    main()
