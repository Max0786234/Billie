"""Assemble the Hugging Face Space folder (Docker build context) from this repository.

    cd backend
    .venv/Scripts/python ../deploy/huggingface/stage.py ../../hf-space
    hf upload <user>/billie ../../hf-space . --repo-type space

The Space gets only what the image needs: backend source, frontend source, the
knowledge base, the bundled samples, and prebuilt indexes (MedQuAD + the 50k
embedded arXiv papers, via ``python -m billie.ingest bundle``). Never copied:
``.env*``, ``data/`` (API keys, sessions), virtualenvs, node_modules, datasets.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".venv", "node_modules", "dist", "*.tsbuildinfo", ".env*", "test-results", "playwright-report"
)


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    for name in ("backend", "frontend", "knowledge", "datasets"):
        shutil.rmtree(out / name, ignore_errors=True)
    shutil.copytree(REPO / "backend" / "src", out / "backend" / "src", ignore=IGNORE)
    shutil.copy2(REPO / "backend" / "pyproject.toml", out / "backend" / "pyproject.toml")
    shutil.copytree(REPO / "frontend", out / "frontend", ignore=IGNORE)
    shutil.copytree(REPO / "knowledge", out / "knowledge", ignore=IGNORE)
    shutil.copytree(REPO / "datasets" / "samples", out / "datasets" / "samples", ignore=IGNORE)
    for name in ("Dockerfile", "start.sh", "README.md"):
        shutil.copy2(HERE / name, out / name)
    (out / ".gitattributes").write_text("*.sqlite filter=lfs diff=lfs merge=lfs -text\n*.npy filter=lfs diff=lfs merge=lfs -text\n")
    if not (out / "index" / "arxiv.sqlite").exists():
        subprocess.run(
            [sys.executable, "-m", "billie.ingest", "bundle", "--out", str(out / "index")],
            cwd=REPO / "backend",
            check=True,
        )
    leaked = [p for p in out.rglob("*") if p.name.startswith(".env") or p.name == "llm_settings.json"]
    if leaked:
        raise SystemExit(f"refusing: secret-looking files staged: {leaked}")
    print(f"staged Space at {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "hf-space")))
