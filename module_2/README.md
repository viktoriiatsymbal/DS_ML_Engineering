# Homework for Module 2. Containerization

## Description

In this homework, I set up a Docker-based development environment for running inference with a pretrained image classification model. The script loads a small sample of images, runs inference, and saves predictions along with accuracy metrics

* Hugging Face pretrained model used: `nateraw/vit-base-beans`
* Dataset used: `beans`

## Project structure

```
.
├── src/
│   └── infer.py
├── outputs/            #generated results
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .dockerignore
├── .gitignore
```

## Usage

### 1. Clone repository

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Configure environment variables

Create `.env` file based on the example:

```bash
cp .env.example .env
```

Then set your user ID and group ID:

```bash
id -u
id -g
```

Update `.env`:

```
UID=your_uid
GID=your_gid
MODEL_NAME=nateraw/vit-base-beans
DATASET_NAME=beans
NUM_SAMPLES=15
```

### 3. Build Docker image

```bash
docker compose build
```

### 4. Run container

```bash
docker compose run --rm dev
```

### 5. Run inference

Inside the container:

```bash
python src/infer.py
```

## Outputs

Results are saved to the `outputs/` folder:

* `predictions.csv` — predictions for each sample
* `summary.json` — experiment summary
* `run_log.txt` — readable log

## How does it work

1. The script loads a dataset from Hugging Face
2. Selects a small subset of samples
3. Loads a pretrained Vision Transformer model
4. Runs inference on each image
5. Saves predictions and calculates accuracy

## Implementation details

- Non-root user is created using UID and GID from `.env` to avoid permission issues when writing files to mounted volumes on the host machine
- The Hugging Face cache is stored locally to speed up repeated runs

## Notes

* You can change model or dataset via `.env`

## Example run

```bash
docker compose up -d --build
docker compose run --rm dev python src/infer.py
```