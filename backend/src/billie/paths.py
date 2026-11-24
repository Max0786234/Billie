"""Single source of truth for filesystem locations.

Every module resolves files through these constants instead of guessing from
``__file__`` or the current working directory, so the API, the evaluation
scripts and the tests behave the same no matter where they are launched from.

Layout (relative to ``REPO_ROOT``)::

    knowledge/              general-mode knowledge base (tracked)
    datasets/samples/       tiny datasets used by tests and as a demo fallback (tracked)
    datasets/medquad/       full MedQuAD corpus (git-ignored, provided locally)
    datasets/arxiv/         full Kaggle arXiv snapshot (git-ignored, provided locally)
    data/                   mutable runtime state: indexes, sessions, uploads (git-ignored)

Set ``BILLIE_HOME`` to point at a different checkout or a mounted volume.
"""

from __future__ import annotations

import os
from pathlib import Path

# backend/src/billie/paths.py -> parents[3] is the repository root.
_DEFAULT_ROOT = Path(__file__).resolve().parents[3]

REPO_ROOT = Path(os.getenv("BILLIE_HOME", _DEFAULT_ROOT)).resolve()

ENV_FILE = REPO_ROOT / ".env"
KNOWLEDGE_DIR = REPO_ROOT / "knowledge"

DATASETS_DIR = REPO_ROOT / "datasets"
SAMPLES_DIR = DATASETS_DIR / "samples"
MEDQUAD_DIR = DATASETS_DIR / "medquad"
MEDQUAD_SAMPLE = SAMPLES_DIR / "medquad_sample.json"
ARXIV_SNAPSHOT = DATASETS_DIR / "arxiv" / "arxiv-metadata-oai-snapshot.json"
ARXIV_SAMPLE = SAMPLES_DIR / "arxiv_cs_sample.jsonl"

DATA_DIR = REPO_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

BACKEND_DIR = REPO_ROOT / "backend"
EVALS_DIR = BACKEND_DIR / "evals"
EVAL_DATASETS_DIR = EVALS_DIR / "datasets"
EVAL_RESULTS_DIR = REPO_ROOT / "docs" / "evaluation"
