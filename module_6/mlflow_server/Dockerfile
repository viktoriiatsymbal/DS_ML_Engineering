FROM python:3.11-slim
WORKDIR /server
RUN pip install --no-cache-dir \
    mlflow==3.0.1 \
    psycopg2-binary==2.9.10
RUN mkdir -p /server/mlruns \
    && chgrp -R 0 /server \
    && chmod -R g+rwX /server
EXPOSE 5000
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "postgresql+psycopg2://mlflow:mlflow@postgres:5432/mlflow", \
     "--default-artifact-root", "mlflow-artifacts:/", \
     "--artifacts-destination", "/server/mlruns", \
     "--serve-artifacts", \
     "--workers", "1"]