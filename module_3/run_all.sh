set -e

LOCAL_UID=$(id -u)
LOCAL_GID=$(id -g)

echo "Using UID=$LOCAL_UID GID=$LOCAL_GID"

cat > .env <<EOF
LOCAL_UID=$LOCAL_UID
LOCAL_GID=$LOCAL_GID
EOF

echo "Preparing directories..."
mkdir -p mlflow_data artifacts tmp
touch mlflow_data/mlflow.db

echo "Fixing ownership..."
chown -R "$LOCAL_UID:$LOCAL_GID" mlflow_data artifacts tmp

echo "Building Docker images..."
docker compose build

echo "Starting MLflow server..."
docker compose up -d mlflow

echo "Waiting for MLflow API to be ready..."
sleep 15
echo "MLflow is ready!"
echo "Running trainer (training + inference)..."
docker compose up trainer

echo "Done."
echo "MLflow UI: http://localhost:5001"