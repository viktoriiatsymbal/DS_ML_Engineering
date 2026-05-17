from pathlib import Path
import yaml

def test_params_yaml_exists():
    assert Path("params.yaml").exists()

def test_dvc_yaml_exists():
    assert Path("dvc.yaml").exists()

def test_dvc_pipeline_has_required_stages():
    with open("dvc.yaml", "r", encoding="utf-8") as file:
        dvc_config = yaml.safe_load(file)
    stages = dvc_config["stages"]

    assert "preprocess_basic" in stages
    assert "preprocess_scaled" in stages
    assert "preprocess_aggregated" in stages
    assert "train" in stages

def test_dvc_metrics_are_configured():
    with open("dvc.yaml", "r", encoding="utf-8") as file:
        dvc_config = yaml.safe_load(file)
    train_stage = dvc_config["stages"]["train"]

    assert "metrics" in train_stage
    assert "dvclive/metrics.json" in train_stage["metrics"][0]
