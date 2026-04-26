import json
import os
import tempfile
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from src.config import REPORTS_DIR

EXPERIMENT_NAME = "airflow-wine-preprocessing"
REGISTERED_MODEL_NAME = "wine_best_model"
TARGET_COL = "quality_class"

def build_models():
    return [
        (
            "logreg_C0.01",
            LogisticRegression(max_iter=1000, random_state=42, C=0.01),
            {"model_type": "logreg", "C": 0.01},
        ),
        (
            "logreg_C0.1",
            LogisticRegression(max_iter=1000, random_state=42, C=0.1),
            {"model_type": "logreg", "C": 0.1},
        ),
        (
            "logreg_C1.0",
            LogisticRegression(max_iter=1000, random_state=42, C=1.0),
            {"model_type": "logreg", "C": 1.0},
        ),
        (
            "random_forest_depth3",
            RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
            {"model_type": "random_forest", "n_estimators": 100, "max_depth": 3},
        ),
        (
            "random_forest_depth5",
            RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            {"model_type": "random_forest", "n_estimators": 100, "max_depth": 5},
        ),
        (
            "random_forest_depth10",
            RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            {"model_type": "random_forest", "n_estimators": 100, "max_depth": 10},
        ),
        (
            "svc_C0.1",
            SVC(kernel="rbf", C=0.1, probability=True, random_state=42),
            {"model_type": "svc", "kernel": "rbf", "C": 0.1},
        ),
        (
            "svc_C1.0",
            SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
            {"model_type": "svc", "kernel": "rbf", "C": 1.0},
        ),
        (
            "svc_C10.0",
            SVC(kernel="rbf", C=10.0, probability=True, random_state=42),
            {"model_type": "svc", "kernel": "rbf", "C": 10.0},
        ),
    ]

def train_models_with_mlflow(transformed_paths: dict, folds_path: str) -> dict:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(
            EXPERIMENT_NAME,
            artifact_location="mlflow-artifacts:/")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train = pd.read_csv(transformed_paths["X_train"])
    X_test = pd.read_csv(transformed_paths["X_test"])
    y_train = pd.read_csv(transformed_paths["y_train"])[TARGET_COL]
    y_test = pd.read_csv(transformed_paths["y_test"])[TARGET_COL]

    with open(folds_path, "r", encoding="utf-8") as f:
        folds = json.load(f)

    results = []
    best_score = -1
    best_run_id = None
    best_model_name = None

    for model_name, model, params in build_models():
        with mlflow.start_run(run_name=model_name):
            mlflow.set_tag("project", "airflow-homework")
            mlflow.set_tag("stage", "training")
            mlflow.set_tag("preprocessing", "fitted_on_train_only")

            for key, value in params.items():
                mlflow.log_param(key, value)

            fold_f1_scores = []
            fold_accuracy_scores = []

            for fold in folds:
                train_idx = fold["train_idx"]
                val_idx = fold["val_idx"]

                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]

                fold_model = clone(model)
                fold_model.fit(X_fold_train, y_fold_train)

                preds = fold_model.predict(X_fold_val)

                fold_f1_scores.append(f1_score(y_fold_val, preds))
                fold_accuracy_scores.append(accuracy_score(y_fold_val, preds))

            mean_cv_f1 = round(float(sum(fold_f1_scores) / len(fold_f1_scores)), 4)
            mean_cv_accuracy = round(
                float(sum(fold_accuracy_scores) / len(fold_accuracy_scores)), 4
            )

            mlflow.log_metric("mean_cv_f1", mean_cv_f1)
            mlflow.log_metric("mean_cv_accuracy", mean_cv_accuracy)

            final_model = clone(model)
            final_model.fit(X_train, y_train)

            test_preds = final_model.predict(X_test)

            test_f1 = round(float(f1_score(y_test, test_preds)), 4)
            test_accuracy = round(float(accuracy_score(y_test, test_preds)), 4)

            mlflow.log_metric("test_f1", test_f1)
            mlflow.log_metric("test_accuracy", test_accuracy)

            with tempfile.TemporaryDirectory() as tmp_dir:
                mlflow.sklearn.save_model(final_model, tmp_dir)
                mlflow.log_artifacts(tmp_dir, artifact_path="model")

            run_id = mlflow.active_run().info.run_id

            result = {
                "run_id": run_id,
                "model_name": model_name,
                "mean_cv_f1": mean_cv_f1,
                "mean_cv_accuracy": mean_cv_accuracy,
                "test_f1": test_f1,
                "test_accuracy": test_accuracy,
            }

            results.append(result)

            if mean_cv_f1 > best_score:
                best_score = mean_cv_f1
                best_run_id = run_id
                best_model_name = model_name

    results_df = pd.DataFrame(results).sort_values("mean_cv_f1", ascending=False)
    results_path = REPORTS_DIR / "experiment_results.csv"
    results_df.to_csv(results_path, index=False)

    best_info = {
        "best_run_id": best_run_id,
        "best_model_name": best_model_name,
        "best_cv_f1": best_score,
        "model_artifact_path": "model",
    }
    best_info_path = REPORTS_DIR / "best_model_info.json"
    with open(best_info_path, "w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=2)
    return best_info


def register_best_model(best_run_info: dict) -> str:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    model_uri = f"runs:/{best_run_info['best_run_id']}/model"
    registered = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME)
    final_model_uri = f"models:/{REGISTERED_MODEL_NAME}/{registered.version}"
    return final_model_uri
