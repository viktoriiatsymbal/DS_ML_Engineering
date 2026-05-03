import sys
from datetime import datetime

sys.path.append("/opt/airflow/project/src")

from airflow.decorators import dag, task
from wine_deployment.batch_image import build_online_docker_image
from wine_deployment.cleaning import clean_data
from wine_deployment.data_download import download_wine_data
from wine_deployment.data_merge import create_extra_source, merge_sources
from wine_deployment.inference import run_batch_inference
from wine_deployment.splitting import create_cv_folds, split_train_test
from wine_deployment.training import register_best_model, train_models_with_mlflow

@dag(
    dag_id="wine_batch_deployment_pipeline",
    start_date=datetime(2026, 5, 2),
    schedule=None,
    catchup=False,
    tags=["module5", "deployment", "batch", "rest"],
)
def wine_batch_deployment_pipeline():

    @task
    def download_task():
        return download_wine_data()

    @task
    def create_extra_source_task(raw_path):
        return create_extra_source(raw_path)

    @task
    def merge_task(raw_path, extra_path):
        return merge_sources(raw_path, extra_path)

    @task
    def clean_task(merged_path):
        return clean_data(merged_path)

    @task
    def split_task(clean_path):
        return split_train_test(clean_path)

    @task
    def cv_task(split_paths):
        return create_cv_folds(split_paths)

    @task
    def train_task(split_paths, folds_path):
        return train_models_with_mlflow(split_paths, folds_path)

    @task
    def register_task(best_run_info):
        return register_best_model(best_run_info)

    @task
    def batch_predict_task(model_uri, split_paths):
        return run_batch_inference(model_uri, split_paths)

    @task
    def build_online_image_task(model_uri):
        return build_online_docker_image(model_uri)

    raw_path = download_task()
    extra_path = create_extra_source_task(raw_path)
    merged_path = merge_task(raw_path, extra_path)
    clean_path = clean_task(merged_path)

    split_paths = split_task(clean_path)
    folds_path = cv_task(split_paths)

    best_run_info = train_task(split_paths, folds_path)
    model_uri = register_task(best_run_info)

    predictions_path = batch_predict_task(model_uri, split_paths)
    image_result = build_online_image_task(model_uri)

    predictions_path >> image_result

wine_batch_deployment_pipeline()
