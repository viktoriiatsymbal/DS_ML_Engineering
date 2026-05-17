import os
import shutil
from pathlib import Path
import docker
import mlflow
from wine_deployment.config import MLFLOW_TRACKING_URI, ONLINE_MODEL_DIR, PROJECT_ROOT, BUILD_DIR

def prepare_online_model_artifact(model_uri):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI))
    temp_download_dir = BUILD_DIR / "_downloaded_model"

    if temp_download_dir.exists():
        shutil.rmtree(temp_download_dir)

    if ONLINE_MODEL_DIR.exists():
        shutil.rmtree(ONLINE_MODEL_DIR)

    temp_download_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri,
            dst_path=str(temp_download_dir)))
    if not downloaded_path.exists():
        raise FileNotFoundError(f"Downloaded model path does not exist: {downloaded_path}")
    mlmodel_file = downloaded_path / "MLmodel"

    if not mlmodel_file.exists():
        possible_model_dirs = [
            path for path in downloaded_path.rglob("MLmodel")
            if path.is_file()]
        if not possible_model_dirs:
            raise FileNotFoundError(
                f"Could not find MLmodel file inside downloaded artifact: {downloaded_path}")
        downloaded_path = possible_model_dirs[0].parent

    shutil.copytree(downloaded_path, ONLINE_MODEL_DIR)
    if temp_download_dir.exists():
        shutil.rmtree(temp_download_dir)
    return str(ONLINE_MODEL_DIR)

def build_online_docker_image(model_uri, image_tag="wine-online:latest"):
    model_path = prepare_online_model_artifact(model_uri)
    client = docker.from_env()
    image, logs = client.images.build(
        path=str(PROJECT_ROOT),
        dockerfile="online.Dockerfile",
        tag=image_tag,
        rm=True)

    log_lines = []
    for line in logs:
        if "stream" in line:
            log_lines.append(line["stream"].strip())
    return (
        f"Prepared model artifact at: {model_path}\n"
        f"Built image: {image_tag}\n"
        + "\n".join(log_lines[-20:]))
