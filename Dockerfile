FROM python:3.15-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

COPY requirements.txt .

RUN python -m venv /opt/venv \
 && grep -vE '^(pytest|#|$)' requirements.txt > requirements-prod.txt \
 && /opt/venv/bin/pip install --only-binary=:all: -r requirements-prod.txt

FROM python:3.15-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=WARNING \
    CACHE_DB_PATH=/app/cache/data_cache.db \
    MPLCONFIGDIR=/home/app/.cache/matplotlib

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_SERVER_RUN_ON_SAVE=false \
    STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=none \
    STREAMLIT_CLIENT_TOOLBAR_MODE=viewer \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_LOGGER_LEVEL=warning

RUN useradd --create-home --uid 1000 app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY run.py ./
COPY src ./src
COPY .streamlit ./.streamlit
COPY data/mock_responses.json ./data/

RUN python -m compileall -q src \
 && mkdir -p /app/cache \
 && chown app:app /app/cache

USER app

RUN python -c "import matplotlib.pyplot"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request as r; b = os.getenv('STREAMLIT_SERVER_BASE_URL_PATH', '').strip('/'); r.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8501') + ('/' + b if b else '') + '/_stcore/health', timeout=4)"

CMD ["sh", "-c", "exec streamlit run run.py --server.port=${PORT:-8501}"]
