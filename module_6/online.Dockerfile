FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY build/online_model /app/model
RUN pip install --no-cache-dir /app \
    && useradd -m -u 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser
ENV MODEL_DIR=/app/model
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=8000
EXPOSE 8000
CMD ["python", "-m", "wine_deployment.app"]