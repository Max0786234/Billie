# Evaluation results

| Folder | What | Reproduce (from `backend/`) |
|---|---|---|
| [`retrieval/`](retrieval) | Medical retrieval on TREC 2017 LiveQA consumer questions, scored against MedQuAD's human relevance judgments. Aggregate scores (`liveqa_medquad.json`) and per-query rankings | `python -m evals.retrieval_liveqa` |
| [`sentiment/`](sentiment) | Multilingual sentiment head: per-language test scores against a word-list baseline | `python -m evals.train_sentiment --per-lang 800` |

The tables and how to read them are in the main [README](../../README.md#evaluation).
The pre-overhaul baseline (a 5-question legacy benchmark) is kept for reference in
[`../archive/evaluation-baseline-2026-09-26`](../archive/evaluation-baseline-2026-09-26).
