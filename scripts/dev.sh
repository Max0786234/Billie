#!/usr/bin/env bash
# Start Billie for local development: FastAPI on :8000 and the React dev server on :5173.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
venv="$root/backend/.venv"
py="$venv/bin/python"; [ -x "$py" ] || py="$venv/Scripts/python.exe"

if [ ! -x "$py" ]; then
  echo "Creating Python 3.12 virtual environment..."
  python3.12 -m venv "$venv"
  py="$venv/bin/python"
  "$py" -m pip install --upgrade pip
  "$py" -m pip install -e "$root/backend[dev]"
fi
[ -d "$root/frontend/node_modules" ] || (cd "$root/frontend" && npm install)
curl -sf http://127.0.0.1:11434/api/version >/dev/null || echo "WARNING: Ollama is not running - answers will be retrieval-only."

(cd "$root/backend" && "$py" -m uvicorn billie.api.app:create_app --factory --reload --port 8000) &
api=$!
trap 'kill $api 2>/dev/null' EXIT
echo "API      http://127.0.0.1:8000/docs"
echo "Billie   http://127.0.0.1:5173"
cd "$root/frontend" && npm run dev
