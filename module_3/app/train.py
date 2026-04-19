import os
import json
import copy
import mlflow
import mlflow.sklearn
import pandas as pd
from joblib import dump
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from utils import ensure_dir, save_confusion_matrix_plot, save_results_plot

EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "wine-model-comparison")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
RANDOM_STATE = 42
TEST_SIZE = 0.2

def build_feature_sets(df):
    all_features = df.columns.tolist()
    return {
        "all_features": all_features,
        "first_6_features": all_features[:6],
        "last_7_features": all_features[6:],
        "alcohol_color_subset": [c for c in all_features if c in [
            "alcohol",
            "color_intensity",
            "hue",
            "od280/od315_of_diluted_wines",
            "proline",
        ]],
    }

def build_models():
    configs = []
    for c in [0.1, 1.0, 10.0]:
        configs.append((
            "logreg",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=c, max_iter=1000, random_state=RANDOM_STATE))
            ]), {"C": c}))
    for n_estimators in [50, 100]:
        for max_depth in [3, 5, None]:
            configs.append((
                "random_forest",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=RANDOM_STATE,
                ), {"n_estimators": n_estimators, "max_depth": max_depth}))
    for c in [0.5, 1.0]:
        for kernel in ["linear", "rbf"]:
            configs.append((
                "svc",
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", SVC(C=c, kernel=kernel, probability=True, random_state=RANDOM_STATE))
                ]), {"C": c, "kernel": kernel}))
    return configs

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    out_dir = ensure_dir("artifacts")

    wine = load_wine(as_frame=True)
    X_full = wine.data.copy()
    y = wine.target.copy()
    class_names = [str(x) for x in wine.target_names]

    feature_sets = build_feature_sets(X_full)
    model_configs = build_models()

    all_results = []
    best_score = -1.0
    best_run_id = None
    best_model = None
    best_feature_columns = None

    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_full, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    with mlflow.start_run(run_name="parent_grid_search"):
        mlflow.set_tag("project", "mlflow-homework")
        mlflow.set_tag("task", "wine_classification")
        mlflow.set_tag("mlflow.note.content", "Model comparison across feature sets and hyperparams")

        for feature_set_name, columns in feature_sets.items():
            X_train = X_train_full[columns]
            X_test = X_test_full[columns]
            for model_name, model, params in model_configs:
                run_name = f"{model_name}_{feature_set_name}"
                with mlflow.start_run(run_name=run_name, nested=True):
                    mlflow.log_param("feature_set", feature_set_name)
                    mlflow.log_param("n_features", len(columns))
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("random_state", RANDOM_STATE)
                    mlflow.log_param("test_size", TEST_SIZE)

                    for k, v in params.items():
                        mlflow.log_param(k, v)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds, average="macro")
                    cm = confusion_matrix(y_test, preds)
                    mlflow.log_metric("accuracy", acc)
                    mlflow.log_metric("f1_macro", f1)

                    cm_path = out_dir / f"cm_{model_name}_{feature_set_name}.png"
                    save_confusion_matrix_plot(
                        cm=cm,
                        labels=class_names,
                        out_path=cm_path,
                        title=f"{model_name} | {feature_set_name}")
                    mlflow.log_artifact(str(cm_path), artifact_path="plots")

                    result_row = {
                        "run_id": mlflow.active_run().info.run_id,
                        "run_name": run_name,
                        "model_name": model_name,
                        "feature_set": feature_set_name,
                        "n_features": len(columns),
                        "accuracy": acc,
                        "f1_macro": f1,
                        **params,
                    }
                    all_results.append(result_row)

                    if f1 > best_score:
                        best_score = f1
                        best_run_id = mlflow.active_run().info.run_id
                        best_model = copy.deepcopy(model)
                        best_feature_columns = columns

        results_df = pd.DataFrame(all_results).sort_values("f1_macro", ascending=False)
        results_csv = out_dir / "all_results.csv"
        results_df.to_csv(results_csv, index=False)
        mlflow.log_artifact(str(results_csv), artifact_path="reports")

        plot_path = out_dir / "top_runs.png"
        save_results_plot(results_df, plot_path)
        mlflow.log_artifact(str(plot_path), artifact_path="plots")

        best_model_path = out_dir / "best_model.joblib"
        dump({
            "model": best_model,
            "features": best_feature_columns
        }, best_model_path)
        mlflow.log_artifact(str(best_model_path), artifact_path="model")

        best_features_path = out_dir / "best_features.json"
        with open(best_features_path, "w", encoding="utf-8") as f:
            json.dump(best_feature_columns, f, indent=2)
        mlflow.log_artifact(str(best_features_path), artifact_path="model")

        summary = {
            "best_run_id": best_run_id,
            "best_f1_macro": float(best_score),
            "best_model_path": str(best_model_path),
            "best_features_path": str(best_features_path),
        }

        summary_path = out_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(str(summary_path), artifact_path="reports")
        print("\nTop experiment results:")
        print(
            results_df[
                ["run_name", "model_name", "feature_set", "accuracy", "f1_macro"]
            ].head(5).to_string(index=False)
        )
        print("\nBest summary:")
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
