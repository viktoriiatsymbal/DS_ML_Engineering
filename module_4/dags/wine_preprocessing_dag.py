import sys
from datetime import datetime

sys.path.append("/opt/airflow")

from airflow.decorators import dag, task
from src.data_download import download_wine_data
from src.data_merge import create_extra_source, merge_sources
from src.cleaning import clean_data
from src.splitting import split_train_test, create_cv_folds
from src.preprocessing import fit_preprocessor, transform_data
from src.training import train_models_with_mlflow, register_best_model
from src.inference import run_inference_from_registry


@dag(
    dag_id="wine_preprocessing_mlflow_pipeline",
    start_date=datetime(2026, 4, 20),
    schedule=None,
    catchup=False,
    tags=["homework", "airflow", "mlflow", "wine"],
)
def wine_pipeline():

    @task
    def download_task():
        return download_wine_data()

    @task
    def create_extra_source_task(raw_path: str):
        return create_extra_source(raw_path)

    @task
    def merge_task(raw_path: str, extra_path: str):
        return merge_sources(raw_path, extra_path)

    @task
    def clean_task(merged_path: str):
        return clean_data(merged_path)

    @task
    def split_task(clean_path: str):
        return split_train_test(clean_path)

    @task
    def cv_task(split_paths: dict):
        return create_cv_folds(split_paths)

    @task
    def fit_preprocessor_task(split_paths: dict):
        return fit_preprocessor(split_paths)

    @task
    def transform_task(split_paths: dict, preprocessor_path: str):
        return transform_data(split_paths, preprocessor_path)

    @task
    def train_task(transformed_paths: dict, folds_path: str):
        return train_models_with_mlflow(transformed_paths, folds_path)

    @task
    def register_task(best_run_info: dict):
        return register_best_model(best_run_info)

    @task
    def inference_task(model_uri: str, transformed_paths: dict):
        return run_inference_from_registry(model_uri, transformed_paths)

    raw_path = download_task()
    extra_path = create_extra_source_task(raw_path)
    merged_path = merge_task(raw_path, extra_path)
    clean_path = clean_task(merged_path)

    split_paths = split_task(clean_path)
    folds_path = cv_task(split_paths)

    preprocessor_path = fit_preprocessor_task(split_paths)
    transformed_paths = transform_task(split_paths, preprocessor_path)

    best_run_info = train_task(transformed_paths, folds_path)
    model_uri = register_task(best_run_info)

    inference_task(model_uri, transformed_paths)

wine_pipeline()
