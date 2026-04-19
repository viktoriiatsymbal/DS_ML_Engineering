import os
import json
import copy
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from utils import ensure_dir, save_confusion_matrix_plot, save_results_plot

EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "wine-model-comparison")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

def build_feature_sets(df):
    all_features = df.columns.tolist()
    return {
        "all_features": all_features,
        "first_6_features": all_features[:6],
        "last_7_features": all_features[6:],
        "alcohol_color_subset": [
            c for c in all_features
            if c in [
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
                ("model", LogisticRegression(C=c, max_iter=1000, random_state=RANDOM_STATE)),
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
                    ("model", SVC(C=c, kernel=kernel, probability=True, random_state=RANDOM_STATE)),
                ]), {"C": c, "kernel": kernel}))
    return configs

def evaluate_with_cv(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_accuracies = []
    fold_f1s = []

    for train_idx, val_idx in skf.split(X, y):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        model_fold = clone(model)
        model_fold.fit(X_train_fold, y_train_fold)
        preds = model_fold.predict(X_val_fold)

        fold_accuracies.append(accuracy_score(y_val_fold, preds))
        fold_f1s.append(f1_score(y_val_fold, preds, average="macro"))

    return {
        "mean_cv_accuracy": round(float(np.mean(fold_accuracies)), 4),
        "mean_cv_f1_macro": round(float(np.mean(fold_f1s)), 4),
    }

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
    best_model_name = None
    best_params = None

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_full, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    with mlflow.start_run(run_name="parent_grid_search_cv"):
        mlflow.set_tag("project", "mlflow-homework")
        mlflow.set_tag("task", "wine_classification")
        mlflow.set_tag("mlflow.note.content",
            "Model comparison across feature sets and hyperparams using StratifiedKFold CV")

        for feature_set_name, columns in feature_sets.items():
            X_train_val_subset = X_train_val[columns]

            for model_name, model, params in model_configs:
                run_name = f"{model_name}_{feature_set_name}"
                with mlflow.start_run(run_name=run_name, nested=True):
                    mlflow.log_param("feature_set", feature_set_name)
                    mlflow.log_param("n_features", len(columns))
                    mlflow.log_param("model_name", model_name)

                    for k, v in params.items():
                        mlflow.log_param(k, v)

                    cv_metrics = evaluate_with_cv(
                        model=model,
                        X=X_train_val_subset,
                        y=y_train_val,
                        n_splits=N_SPLITS)

                    mlflow.log_metric("mean_cv_accuracy", cv_metrics["mean_cv_accuracy"])
                    mlflow.log_metric("mean_cv_f1_macro", cv_metrics["mean_cv_f1_macro"])

                    result_row = {
                        "run_id": mlflow.active_run().info.run_id,
                        "run_name": run_name,
                        "model_name": model_name,
                        "feature_set": feature_set_name,
                        "n_features": len(columns),
                        "mean_cv_accuracy": cv_metrics["mean_cv_accuracy"],
                        "mean_cv_f1_macro": cv_metrics["mean_cv_f1_macro"],
                        **params,
                    }
                    all_results.append(result_row)

                    if cv_metrics["mean_cv_f1_macro"] > best_score:
                        best_score = cv_metrics["mean_cv_f1_macro"]
                        best_run_id = mlflow.active_run().info.run_id
                        best_model = copy.deepcopy(model)
                        best_feature_columns = columns.copy()
                        best_model_name = model_name
                        best_params = params.copy()

        results_df = pd.DataFrame(all_results).sort_values("mean_cv_f1_macro", ascending=False)

        results_csv = out_dir / "all_results.csv"
        results_df.to_csv(results_csv, index=False)
        mlflow.log_artifact(str(results_csv), artifact_path="reports")

        plot_df = results_df.rename(columns={"mean_cv_f1_macro": "f1_macro"})
        plot_path = out_dir / "top_runs.png"
        save_results_plot(plot_df, plot_path)
        mlflow.log_artifact(str(plot_path), artifact_path="plots")

        X_trainval_best = X_train_val[best_feature_columns]
        X_test_best = X_test[best_feature_columns]

        best_model.fit(X_trainval_best, y_train_val)
        test_preds = best_model.predict(X_test_best)

        test_acc = round(accuracy_score(y_test, test_preds), 4)
        test_f1 = round(f1_score(y_test, test_preds, average="macro"), 4)
        test_cm = confusion_matrix(y_test, test_preds)

        mlflow.log_metric("final_test_accuracy", test_acc)
        mlflow.log_metric("final_test_f1_macro", test_f1)

        final_cm_path = out_dir / "cm_best_model_test.png"
        save_confusion_matrix_plot(
            cm=test_cm,
            labels=class_names,
            out_path=final_cm_path,
            title="Best model | final test",
        )
        mlflow.log_artifact(str(final_cm_path), artifact_path="plots")

        best_model_path = out_dir / "best_model.joblib"
        dump(
            {
                "model": best_model,
                "features": best_feature_columns,
            },
            best_model_path)
        mlflow.log_artifact(str(best_model_path), artifact_path="model")

        best_features_path = out_dir / "best_features.json"
        with open(best_features_path, "w", encoding="utf-8") as f:
            json.dump(best_feature_columns, f, indent=2)
        mlflow.log_artifact(str(best_features_path), artifact_path="model")

        summary = {
            "best_run_id": best_run_id,
            "best_mean_cv_f1_macro": round(float(best_score), 4),
            "final_test_accuracy": round(float(test_acc), 4),
            "final_test_f1_macro": round(float(test_f1), 4),
            "best_model_name": best_model_name,
            "best_params": best_params,
            "best_feature_columns": best_feature_columns,
            "best_model_path": str(best_model_path),
            "best_features_path": str(best_features_path),
        }

        summary_path = out_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(str(summary_path), artifact_path="reports")

        print("\nTop CV experiment results:")
        print(
            results_df[
                [
                    "run_name",
                    "model_name",
                    "feature_set",
                    "mean_cv_accuracy",
                    "mean_cv_f1_macro",
                ]
            ].head(5).to_string(index=False)
        )

        print("\nFinal test evaluation:")
        print(json.dumps(
            {
                "final_test_accuracy": float(test_acc),
                "final_test_f1_macro": float(test_f1),
            },
            indent=2
        ))

        print("\nBest summary:")
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
