---
title: Billie
emoji: 🩺
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Cited RAG answers over MedQuAD, arXiv and docs
---

# Billie: a retrieval-augmented assistant that cites its sources

**Public demo.** Ask questions in three modes:

| Mode | Searches |
|---|---|
| General | Billie's own documentation |
| Medical | MedQuAD: 27.6k passages from NIH sources such as MedlinePlus, NIDDK and NCI |
| Research | 50,000 recent arXiv computer-science papers |

Every answer is built only from retrieved passages and cites them as `[1]`, `[2]`… Billie checks the citations afterwards, and replies plainly when the sources don't cover a question. It understands English, Hindi, Hinglish, Spanish, French and German. You can also attach an image.

**How this demo runs:**
- Answers are written by `qwen/qwen3.8-27b` on Groq, and images are read by Gemini 3.5 Flash-Lite.
- Search is hybrid: BM25 (SQLite FTS5) and dense `embeddinggemma` vectors, fused with Reciprocal Rank Fusion. The embedding model runs inside this container.
- The demo runs on free API quotas, so each visitor gets a limited number of questions per day.
- Your chats are private to your browser and are deleted when the Space restarts.

Billie isn't a doctor. Medical answers are for information only.

**Source code, evaluation results and instructions to run it fully offline with local models:**
https://github.com/Vishaljhaaa/Billie
