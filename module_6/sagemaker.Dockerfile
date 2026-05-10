FROM python:3.11-slim
WORKDIR /opt/program
COPY requirements.txt /opt/program/requirements.txt
RUN pip install --no-cache-dir -r /opt/program/requirements.txt
COPY pyproject.toml /opt/program/pyproject.toml
COPY src /opt/program/src
RUN pip install --no-cache-dir /opt/program
ENV MODEL_DIR=/opt/ml/model
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=8080
EXPOSE 8080
ENTRYPOINT ["python", "-m", "wine_deployment.sagemaker_app"]