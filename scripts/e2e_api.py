"""Start an API for end-to-end tests: real indexes, local Ollama models, throwaway state.

    cd backend
    .venv/Scripts/python ../scripts/e2e_api.py --port 8020
    # in frontend/: BILLIE_API_URL=http://127.0.0.1:8020 npx vite --port 5175
    #               BILLIE_UI_URL=http://127.0.0.1:5175 npx playwright test

The app's saved model settings (which may point at a cloud model) and its chat history,
answer cache and uploads are left untouched: they are replaced by files in a temp folder,
so tests never spend cloud quota and never mix test chats into real history.
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import replace
from pathlib import Path

import uvicorn

from billie.api.app import create_app
from billie.runtime import load_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--demo", action="store_true", help="public demo mode (private chats, quotas, own documents)")
    args = parser.parse_args()
    base = load_settings()
    tmp = Path(tempfile.mkdtemp(prefix="billie-e2e-"))
    settings = replace(
        base,
        demo=args.demo,
        rate_limit_per_minute=0,
        paths=replace(
            base.paths,
            data_dir=tmp,  # answer cache and private documents
            sessions_db=tmp / "sessions.sqlite",
            llm_settings=tmp / "llm_settings.json",  # absent: defaults to local Ollama
            image_uploads_dir=tmp / "images",
        ),
    )
    print(f"e2e API on :{args.port} (local models, state in {tmp})", flush=True)
    uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
