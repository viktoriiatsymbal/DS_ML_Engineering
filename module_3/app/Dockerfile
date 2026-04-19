FROM python:3.11-slim

ARG LOCAL_UID=1000
ARG LOCAL_GID=1000
RUN if ! getent group ${LOCAL_GID}; then \
        groupadd -g ${LOCAL_GID} appgroup; \
    fi && \
    useradd -m -u ${LOCAL_UID} -g ${LOCAL_GID} appuser

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY app /app
RUN chmod +x /app/run.sh
CMD ["/bin/bash", "/app/run.sh"]