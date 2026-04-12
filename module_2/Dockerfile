FROM python:3.11-slim

ARG UID=1000
ARG GID=1000
ARG USERNAME=appuser

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN if getent group ${GID} >/dev/null; then \
        useradd --no-log-init --create-home --uid ${UID} --gid ${GID} --shell /bin/bash ${USERNAME}; \
    else \
        groupadd --gid ${GID} ${USERNAME}; \
        useradd --no-log-init --create-home --uid ${UID} --gid ${GID} --shell /bin/bash ${USERNAME}; \
    fi

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY --chown=${UID}:${GID} . .

USER ${USERNAME}

CMD ["bash"]