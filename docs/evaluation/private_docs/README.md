# Questions about visitors' own documents

| File | What |
|---|---|
| `results_local.json`, `summary_local.json` | Final run, local qwen3:4b-instruct, 30 questions: 28/30 |
| `results_groq_spotcheck.json`, `summary_groq_spotcheck.json` | Groq spot-check on the 10 hardest questions: 9/10 (U03 re-scored after a scorer fix; answers unchanged) |
| `*_before_fixes.json` | First local run (25/30), before the passage-id and chunk-collapse fixes (DEVLOG P2.20) |

Re-run: `cd backend && python -m evals.private_docs_eval` (local model; `--use-app-model` for the configured cloud model).
