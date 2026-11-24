# syntax=docker/dockerfile:1
# ---- 1. Build the React frontend ---------------------------------------------------
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- 2. Python runtime serving the API and the built frontend ---------------------
FROM python:3.12-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    BILLIE_HOME=/app BILLIE_DATA_DIR=/app/data
WORKDIR /app
COPY backend/pyproject.toml backend/pyproject.toml
COPY backend/src backend/src
RUN pip install ./backend[anthropic] && useradd --create-home --uid 1000 billie
COPY knowledge knowledge
COPY datasets/samples datasets/samples
COPY datasets/README.md datasets/README.md
COPY --from=web /web/dist frontend/dist
RUN mkdir -p data && chown -R billie:billie /app
USER billie
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1
CMD ["uvicorn", "billie.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
