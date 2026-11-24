# Stress test results

| File | What |
|---|---|
| `REPORT.md`, `results.json`, `summary.json` | Full run: 96 questions + 4 cache repeats on the local model (`qwen3:4b-instruct`), before the fixes described in `docs/DEVLOG.md` (P2.11) |
| `*_groq_before.json`, `*_groq_after.json` | 8 questions on the live Groq setup, before/after the speed work (P2.13) |
| `*_local_after.json` | 4 questions on the local model after the speed work |
| `*_groq_full.json` | **Full run on the default cloud setup** (Groq text + Gemini images): 100/100 after re-scoring refusal-with-citation answers as abstained (P2.16); mean 2.0 s |

Re-run: `cd backend && python -m evals.stress_test` (local model; add `--use-app-model` for the configured cloud model).
