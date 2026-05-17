# HW7: Data governance

This project adds DVC data versioning and reproducible pipelines to the previous Wine Quality homework

## Setup

```bash
cd module_7
```

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
```

## DVC remote

The project uses Google Drive as DVC remote storage

```bash
python -m dvc remote list
```

To run `dvc pull` from another machine, you must have access to the Google Drive remote folder. The custom Google OAuth app is in testing mode, so you need to be added as a test user

## Restore data and artifacts

```bash
python -m dvc pull
```

## Reproduce pipeline

```bash
python -m dvc repro
```

To force all stages to rerun:

```bash
python -m dvc repro --force
```

## Show pipeline graph

```bash
python -m dvc dag
```

## Show metrics

```bash
python -m dvc metrics show
```

## Run tests

```bash
pytest -v tests/test_dvc_pipeline_config.py
```

## Pipeline

`preprocess_basic` - cleans raw data and creates `quality_class` and `alcohol_segment`

`preprocess_scaled` - scales numeric features

`preprocess_aggregated` - creates aggregated statistics

`train` - reuses the existing MLflow training logic and saves DVC metrics

## Verification screenshots

Screenshots are stored in:

```text
screenshots/
```