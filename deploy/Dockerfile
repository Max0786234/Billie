# syntax=docker/dockerfile:1
# Billie as one container (live on Azure Container Apps; also runs as a Hugging Face Docker Space).
#
# * Answers come from Groq (text) and Gemini (images); their keys are Space secrets.
# * Embeddings (hybrid search, citation repair, sentiment) run locally in the container
#   with Ollama + embeddinggemma - the same model the indexes were built with.
# * Indexes are prebuilt (`python -m billie.ingest bundle`): MedQuAD + 50k arXiv papers.
#
# Build context = the folder made by deploy/stage.py.

# ---- 1. React frontend ------------------------------------------------------------
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- 2. Ollama runtime + the embedding model (pulled at build time) ----------------
FROM ollama/ollama:0.40.17 AS ollama
ENV OLLAMA_MODELS=/models
RUN ollama serve >/tmp/ollama.log 2>&1 & \
    for i in $(seq 1 60); do ollama list >/dev/null 2>&1 && break; sleep 1; done; \
    ollama pull embeddinggemma
# GPU back-ends are useless on a CPU Space and add gigabytes.
RUN rm -rf /usr/lib/ollama/cuda* /usr/lib/ollama/rocm* /usr/lib/ollama/vulkan* || true

# ---- 3. Python API serving everything on port 7860 ---------------------------------
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* && useradd --create-home --uid 1000 user
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    BILLIE_HOME=/app BILLIE_DATA_DIR=/app/data \
    OLLAMA_MODELS=/app/ollama-models OLLAMA_HOST=127.0.0.1:11434 OLLAMA_KEEP_ALIVE=-1 \
    OLLAMA_BASE_URL=http://127.0.0.1:11434 \
    BILLIE_ENV=production BILLIE_DEMO=1 \
    BILLIE_LLM_PROVIDER=groq BILLIE_LOCAL_FALLBACK=0 \
    BILLIE_VISION_MODEL="" BILLIE_VISION_CLOUD=gemini \
    BILLIE_FRAME_ANCESTORS=https://huggingface.co \
    BILLIE_RATE_LIMIT_PER_MINUTE=12 BILLIE_MAX_IMAGE_MB=4
WORKDIR /app
COPY --from=ollama /usr/bin/ollama /usr/bin/ollama
COPY --from=ollama /usr/lib/ollama /usr/lib/ollama
COPY backend/pyproject.toml backend/pyproject.toml
COPY backend/src backend/src
RUN pip install ./backend
COPY knowledge knowledge
COPY datasets/samples datasets/samples
COPY --from=web /web/dist frontend/dist
COPY --from=ollama --chown=1000:1000 /models ollama-models
COPY --chown=1000:1000 index data/index
COPY start.sh start.sh
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh && chown -R 1000:1000 /app/data
USER user
ENV HOME=/home/user
EXPOSE 7860
CMD ["/app/start.sh"]
