import json
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from wine_deployment.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME, REPORTS_DIR, TARGET_COL
from wine_deployment.preprocessing import build_preprocessor

def build_models():
    return [
        (
            "logreg_C0.01",
            LogisticRegression(max_iter=1000, solver="liblinear", random_state=42, C=0.01),
            {"model_type": "logreg", "C": 0.01},
        ),
        (
            "logreg_C0.1",
            LogisticRegression(max_iter=1000, solver="liblinear", random_state=42, C=0.1),
            {"model_type": "logreg", "C": 0.1},
        ),
        (
            "logreg_C1.0",
            LogisticRegression(max_iter=1000, solver="liblinear", random_state=42, C=1.0),
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

def make_pipeline(model, feature_columns):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_columns)),
            ("model", model),
        ]
    )

def train_models_with_mlflow(split_paths, folds_path):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(
            EXPERIMENT_NAME,
            artifact_location="mlflow-artifacts:/")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train = pd.read_csv(split_paths["X_train_raw"])
    X_test = pd.read_csv(split_paths["X_test_raw"])
    y_train = pd.read_csv(split_paths["y_train"])[TARGET_COL]
    y_test = pd.read_csv(split_paths["y_test"])[TARGET_COL]

    with open(folds_path, "r", encoding="utf-8") as f:
        folds = json.load(f)

    feature_columns = list(X_train.columns)
    results = []
    best_score = -1.0
    best_run_id = None
    best_model_name = None

    for model_name, model, params in build_models():
        with mlflow.start_run(run_name=model_name):
            mlflow.set_tag("project", "module5-model-deployment")
            mlflow.set_tag("artifact_type", "full_pipeline_preprocessor_plus_model")

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

                pipeline = make_pipeline(clone(model), feature_columns)
                pipeline.fit(X_fold_train, y_fold_train)
                preds = pipeline.predict(X_fold_val)
                fold_f1_scores.append(f1_score(y_fold_val, preds))
                fold_accuracy_scores.append(accuracy_score(y_fold_val, preds))

            mean_cv_f1 = round(float(sum(fold_f1_scores) / len(fold_f1_scores)), 4)
            mean_cv_accuracy = round(
                float(sum(fold_accuracy_scores) / len(fold_accuracy_scores)), 4,
            )

            final_pipeline = make_pipeline(clone(model), feature_columns)
            final_pipeline.fit(X_train, y_train)

            test_preds = final_pipeline.predict(X_test)

            test_f1 = round(float(f1_score(y_test, test_preds)), 4)
            test_accuracy = round(float(accuracy_score(y_test, test_preds)), 4)

            mlflow.log_metric("mean_cv_f1", mean_cv_f1)
            mlflow.log_metric("mean_cv_accuracy", mean_cv_accuracy)
            mlflow.log_metric("test_f1", test_f1)
            mlflow.log_metric("test_accuracy", test_accuracy)

            input_example = X_test.head(3)
            signature = infer_signature(
                input_example,
                final_pipeline.predict(input_example))

            model_info = mlflow.sklearn.log_model(
                sk_model=final_pipeline,
                name="model",
                signature=signature,
                input_example=input_example)

            run_id = mlflow.active_run().info.run_id

            result = {
                "run_id": run_id,
                "model_uri": model_info.model_uri,
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

def register_best_model(best_run_info):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI))
    model_uri = f"runs:/{best_run_info['best_run_id']}/model"
    registered = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME)
    final_model_uri = f"models:/{REGISTERED_MODEL_NAME}/{registered.version}"
    model_uri_path = REPORTS_DIR / "model_uri.txt"
    model_uri_path.write_text(final_model_uri, encoding="utf-8")
    return final_model_uri
