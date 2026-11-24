# Deployment

One container serves Billie's API and the built React app on port 7860. Text answers come from Groq and images are read by Gemini, with keys supplied as host secrets. Embeddings run inside the container with Ollama and `embeddinggemma`, the same model the indexes were built with. Demo mode is on (private chats, daily caps, visitors' own temporary documents).

| File | Purpose |
|---|---|
| `Dockerfile` | Three stages: React build → Ollama with `embeddinggemma` pulled at build time → Python 3.12 API (uid 1000, port 7860) |
| `start.sh` | Starts Ollama, waits until it answers, then runs uvicorn with proxy headers so rate limits see the real visitor IP |
| `stage.py` | Assembles the build context: source code, knowledge base, samples and prebuilt indexes. Refuses to include `.env*` or `llm_settings.json` |
| `space_README.md` | The card for a Hugging Face Docker Space, if the image is hosted there instead (Docker Spaces need a paid plan) |

## Live demo: Azure Container Apps

1. **Indexes:** `python -m billie.ingest bundle --out <dir>` writes MedQuAD and the 50k embedded arXiv papers (~300 MB). They're attached to the `index-v1` GitHub release of the repository that builds the image.
2. **Image:** `.github/workflows/build-image.yml` (manual) runs `deploy/stage.py`, builds the Dockerfile and pushes it to Azure Container Registry. Azure's own image builder is disabled on student subscriptions. The registry password is the `ACR_PASSWORD` Actions secret.
3. **App:** Azure Container Apps, 2 vCPU / 4 GiB, 0–1 replicas (switched off when idle), external HTTPS ingress on port 7860.
   - Secrets: `GROQ_API_KEY`, `GEMINI_API_KEY`, `BILLIE_ADMIN_KEY`.
   - Redeploy: `az containerapp update -g billie-rg -n billie --image <registry>/billie:<sha>`.

Environment defaults baked into the image: `BILLIE_ENV=production`, `BILLIE_DEMO=1`, `BILLIE_LLM_PROVIDER=groq`, `BILLIE_LOCAL_FALLBACK=0`, `BILLIE_VISION_CLOUD=gemini`. See `.env.example` for every setting.
