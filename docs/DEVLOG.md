# Billie development log

A chronological engineering record: what was built, what went wrong, what changed, and why. Each entry follows **problem → change → why → evidence**. It is the source material for the full project write-up.

Hardware for all measurements: AMD Ryzen 7 5800HS (8 cores), 16 GB RAM, integrated Radeon graphics (no usable GPU), Windows 11. Ollama 0.40.

---

## Phase 0: audit of the starting point (2026-11-23)

### What existed
- GitHub repo `Vishaljhaaa/Billie` plus a local `project (2).zip` (1.9 GB compressed, 5.9 GB unpacked).
- **The zip was newer than GitHub.** It held uncommitted work on top of the last pushed commit: `app/retrieval.py` (an in-memory BM25), `docs/IMPLEMENTATION_PLAN.md`, `PROJECT_TECHNICAL_LOG.md`, retrieval tests and an evaluation baseline. **Decision:** the zip is the source of truth.
- What took the space: 5.17 GB was one file, the public Kaggle arXiv metadata snapshot; 424 MB was two broken virtualenvs (one pointed at an uninstalled Python, the other was an empty Python 3.14); 48 MB was MedQuAD. The actual code was about 1 MB.
- The code was a FastAPI + Streamlit chatbot with three modes, ~3,600 lines. Baseline test run: **35 passed**.

### Problems found
| Problem | Impact |
|---|---|
| No LLM configured (`OPENAI_API_KEY` empty) | Every "answer" was a pasted retrieved snippet; Billie generated nothing |
| General knowledge base = 3 text files, 1,235 bytes, 11 sentences, all about the chatbot itself | General mode had nothing real to answer from |
| Retrieval = lexical term-frequency cosine | No semantic matching; the planned BM25/hybrid work was unfinished |
| Rule-based "reasoning", e.g. a phishing question → regex for an email address | Fragile special cases presented as intelligence |
| "Multimodal" benchmark referenced images that did not exist | Claimed capability unverified |
| "Multilingual" evaluation set was English only; translation was word substitution | Claimed capability unverified |
| Sentiment = small word lists, scored on 10 examples | Weak and barely measured |
| `/chat` accepted arbitrary server file paths for images | Security hole: any server file could be read and sent to a remote model |
| Auth silently disabled when keys were unset | Fail-open deployment |
| arXiv loader read only the first 5,000 CS papers | Under 1% of the available data |
| 6 overlapping READMEs, 4 requirements files, no lint or type checks | Messy for reviewers |

### Owner decisions (asked up front)
- **LLM:** Ollama, local and free, plus a UI to add a cloud API key later. (Later clarified: no API keys are available, so everything must work locally.)
- **Frontend:** React + FastAPI (replacing Streamlit).
- **Git:** local only for now; no push.
- **Scope:** keep every feature and make each one real, rather than dropping the weak ones.

---

## Step 1: clean structure

**Change:** a new `billie/` layout: `backend/` (`src/billie` package, tests, evals), `frontend/`, `knowledge/`, `datasets/` (samples tracked, full data ignored), `data/` (runtime, ignored), `docs/`. One `pyproject.toml` replaced the 4 requirements files; ruff, mypy and pytest were configured; a CI workflow was added; git history was preserved by copying `.git`.

**Why:** a reviewer must understand the repo in 30 seconds; src-layout avoids accidental imports; a single `paths.py` removes the 6 places that guessed paths from `__file__` or the working directory.

**Problems along the way:**
- *Medical fallback stopped loading after the move.* The sample loader scanned the sample's folder for files named `sample_medquad*`. Fixed by loading the given file directly. Fragile name-based discovery removed.
- *mypy would not run:* "An Application Control policy has blocked this file". Windows Smart App Control blocks mypy's compiled DLLs and its `librt` dependency. Fixed by installing **mypy 1.13 from source** (the last release without `librt`); the pin is documented in `pyproject.toml`.
- First lint run: 57 findings (35 auto-fixed), including a real file-handle leak (`open(path).read()` without closing). First mypy run: 10 real type errors (e.g. a `ServiceContainer | None` used without narrowing).

**Evidence:** 35/35 tests passing after the move (same as baseline); ruff and mypy clean.

---

## Step 2: real LLM layer

### Model choice (measured, not assumed)
| Model | Prompt tok/s | Gen tok/s | Notes |
|---|---|---|---|
| qwen3:8b | 18 | 5.1 | Good quality, too slow on CPU |
| qwen3:4b (installed) | 35–39 | 9.7 | **Thinking-only variant**: always reasons aloud, even with `think:false` and `/no_think`. Unusable for chat |
| llama3.2 3B | 51 | 12.4 | Fastest, but no Bengali support |
| qwen3:4b-instruct (pulled) | ~39 | ~10 | Non-thinking, 100+ languages → **default chat model** |
| gemma3:4b (pulled) | – | – | Vision-capable → **image model** |

**Insight:** on CPU, *prompt processing* dominates time-to-first-token (1,500 prompt tokens ≈ 40 s on the 4B model). That drove every later context-budget decision.

### Design
- `LLMProvider` protocol: `stream`, `list_models`, `health`, `supports_vision`. Implementations: Ollama (NDJSON), OpenAI-compatible (SSE: OpenAI, Groq, Gemini, OpenRouter, LM Studio), Anthropic (official SDK, `claude-opus-5-5`, server-side refusal fallbacks).
- `LLMManager`: runtime-switchable settings saved to `data/llm_settings.json` with atomic writes, one key per provider, keys masked in every API response.
- `strip_think` stream filter removes `<think>…</think>` blocks, even when a tag is split across chunks.

**Why a protocol:** the pipeline never knows which provider runs, so switching from local to cloud is a settings change, not a code change.

**Interruption:** a power cut stopped work mid-step. All files were safe; model downloads resumed. This motivated crash-safe writes everywhere later (atomic index builds, resumable embeddings).

---

## Step 3: retrieval

### Storage decision: SQLite FTS5 for BM25
Alternatives: an in-memory inverted index (several GB of RAM for 980k papers) or Elasticsearch (an extra server). FTS5 ships with Python, ranks with BM25 on disk, and supports Porter stemming. **Measured:** 5–25 ms per query on MedQuAD; 0.35–0.6 s on 980k arXiv papers.

### Dense + hybrid
`embeddinggemma` (768-d, multilingual) via Ollama; L2-normalised NumPy matrix; cosine similarity = one matrix-vector product (3.5 ms over 20k vectors). Fusion with **Reciprocal Rank Fusion** (k=60), because RRF uses ranks and never needs BM25 scores and cosine similarities on one scale. If the embedder is down, search degrades to BM25 with `degraded: true`.

### Data problems found by a strict index
The first MedQuAD build stored 22,418 of 27,582 passages: `INSERT OR IGNORE` was silently dropping rows. Switching to a strict `INSERT` exposed three upstream data problems, one at a time:
1. MedQuAD reuses the same `qid` for *different* questions inside one document (3,204 cases).
2. Document IDs repeat across the 12 subset folders (`0000001` exists in ten of them).
3. CancerGov reuses one document ID across files about different diseases.

**Fix:** passage ID = subset number + file name + QA position. **Result:** all 27,582 passages, 16,407 answered questions (matching the published count once the copyright-removed subsets are excluded). The original code would have lost about 18% of MedQuAD this way.

The arXiv build then failed with the same strict check: the snapshot contains **duplicate papers**. They're deduplicated while streaming. **Result: 980,539 CS papers** (up from 5,000).

### Robustness
- **Atomic builds:** write `<name>.sqlite.building`, then `os.replace`. A failed build no longer leaves a 1.9 GB temp file behind (that happened once).
- **Windows:** a file that is open or memory-mapped cannot be replaced, so readers are closed before a rebuild and vectors load into RAM instead of `mmap`.
- **Resumable embedding:** parts of 512 vectors written atomically; reruns skip finished parts.

### Embedding budget (CPU: 5 passages/s full text, 8/s at 400 chars)
Full MedQuAD would take ~91 min. Embedding only the **first chunk of each answer** (title + 400 chars) took 34 min, while BM25 still covers every chunk. arXiv: the newest 20,000 papers (~40 min).

### General knowledge base
The 11 legacy sentences moved to `evals` as fixtures. General mode became "chat with your documents": 4 real help pages about Billie plus user uploads (PDF/MD/TXT), rebuilt only when a content fingerprint changes.

---

## Step 2/3 integration: the chat pipeline

The rule-based chatbot was replaced by `ChatPipeline.run()`, a generator of SSE events: status → meta (sources, analysis) → tokens → done.

**Context budget:** 4–5 passages × 650 chars, the last 3 turns of history with answers clipped to 500 chars. The **system prompt is constant**, so Ollama reuses its processed prefix; per-request text goes in the last user message.

**Verification semantics:** citations are parsed. Cites only real sources → grounded; cites a non-existent number → ungrounded; cites nothing → abstained (added later, see below).

**SQLite conversation store (WAL)** replaced the JSON memory file, which rewrote everything on each turn, wasn't crash-safe and grew without bound.

### First real end-to-end runs
- "What are the symptoms of glaucoma?" (medical): grounded, cited [3][4], first token 15.5 s, total 29 s.
- Research question on RAG hallucination: 5 relevant recent papers, grounded, total 43 s.
- **Hindi question failed twice.** (a) Keyword search can't match Hindi text to English documents, so it found no sources. (b) Without sources, the model answered from memory and fell into a **repetition loop for 79 s**. Fixes: a short **query translation** call for non-English questions; Ollama `repeat_penalty=1.15`; a **loop guard** that stops a stream when a 3–60 character phrase repeats 5 times. Re-test: translated query → sources found → grounded Hindi answer citing [3].

---

## Step 4: real language and sentiment

- **Language:** script rules (Devanagari → Hindi, Bengali, CJK, …) → romanised-Hindi cue words → py3langid (naive Bayes, 97 languages) → short or low-confidence text defaults to English. 13/13 hand-checked cases correct, including "hi", "BM25?" and "mujhe sir dard ho raha hai".
- **Sentiment:** a 3-class softmax head (NumPy, class-weighted, L2) on `embeddinggemma` embeddings, trained on `cardiffnlp/tweet_sentiment_multilingual` (800/language × 8). It reuses the search model, so there's no extra download and about 30 ms per message.
  - *Problem:* the Hugging Face datasets API returned HTTP 429 mid-download. **Fix:** exponential back-off honouring `Retry-After`, plus a `--fetch-only` mode.
  - **Results (6,960 held-out tweets):** 63.6% accuracy, 0.632 macro-F1, against ~35% for the word-list baseline. The tiny 10-message support set favours the word list (8/10 vs 7/10); reported, not tuned away.

## Step 5: security

| Issue | Fix |
|---|---|
| Arbitrary server path for images | Uploads validated by magic bytes and size, random IDs, ID-only references |
| Fail-open auth | `BILLIE_ENV=production` refuses to start without keys; constant-time comparison |
| SSRF via URL ingestion | URL ingestion removed |
| Prompt injection | Sources fenced in `<sources>`, declared untrusted; **fence tags inside documents are neutralised**. Found while writing tests: a document containing `</sources>` could have closed the block |
| Flooding | Token-bucket rate limit on POST chat, uploads and model tests |
| Browser | CSP, frame-deny, nosniff, no-referrer |

**Found by a test:** once the frontend was built, an unknown `/api/...` path returned 200 from the single-page-app fallback. That was harmless, but misleading, so `/api/*` now always returns 404.

## Step 6: React frontend

Vite + React 19 + TypeScript + Tailwind 4. Features: streaming answers, stage status ("Searching MedQuAD…"), clickable citation chips that highlight source cards, image upload (button, paste, drop), mode switcher, history with rename/delete, the Settings dialog (provider, key, model, test connection), Documents and Status dialogs, dark mode, `?c=<id>` deep links. A hand-written SSE parser, because `EventSource` can't POST, tested with chunk splits at arbitrary bytes and split multi-byte characters.

- *Problem:* a strict CSP can break React builds. Verified by screenshotting the FastAPI-served build in headless Edge.
- *Problem:* npm `ECONNRESET` while model downloads saturated the network. A retry fixed it.

## Step 7: evaluation

**Benchmark choice:** instead of writing questions ourselves (biased), we used the **TREC 2017 LiveQA Medical test questions** (real consumer messages) with the **MedQuAD authors' human relevance judgments** (2,479 graded answers).

**Honesty constraints:** three MedQuAD subsets have no answers for copyright reasons, so only the 39 questions with a grade-3/4 answer in our index are scored. Unjudged answers count as non-relevant (TREC convention). No tuning on these questions.

| Queries | Retriever | nDCG@10 | Recall@10 | MRR@10 | Success@5 |
|---|---|---|---|---|---|
| Original | BM25 | 0.309 | 0.404 | 0.339 | 0.487 |
| Original | Dense | 0.451 | 0.656 | 0.416 | 0.564 |
| Original | Hybrid | 0.451 | 0.562 | 0.416 | 0.615 |
| Paraphrase | BM25 | 0.423 | 0.559 | 0.433 | 0.564 |
| Paraphrase | Dense | 0.471 | 0.712 | 0.469 | 0.744 |
| Paraphrase | Hybrid | 0.537 | 0.743 | 0.514 | 0.795 |

Latency, measured without background load: BM25 p50 12–18 ms; hybrid p50 55–68 ms, p95 81–178 ms. An earlier run showed 3.5 s, because embedding jobs were occupying Ollama. That's a lesson in measuring under clean conditions.

**Image question fix:** the vision model's description started with boilerplate ("Here's a description…"), which polluted the search query. Now it ends with `Search terms:`, which drives retrieval, and the boilerplate is stripped. An honest "not found" answer had been flagged "ungrounded"; the new **abstained** state fixed that.

## Step 8: packaging

`.env.example` (every setting), `scripts/dev.ps1` / `dev.sh` one-command start, a multi-stage Dockerfile (Node build → Python runtime serving the API and the SPA), `docker-compose.yml` with an Ollama service and model pull, CI for backend and frontend, README, `docs/ARCHITECTURE.md`. **Not verified:** Docker isn't installed on this laptop.

**Cleanup:** legacy modules deleted (old chatbot, rule-based reasoning and validation, JSON memory, Chroma store, NLLB translator, Streamlit UI, legacy benchmarks). Dependencies went from 13 runtime packages plus 4 requirements files to 10 runtime packages: NLTK, matplotlib and BeautifulSoup were removed. Ruff's SIM905 had expanded word lists into 200-line lists; rule disabled and the compact form restored.

---

## Post-launch fixes from owner feedback

### Hinglish question answered badly (2026-11-23)
**Report:** "madhumesh hota kya h" ("what is diabetes", with a typo) got a Devanagari reply, a list of *symptoms* instead of a definition, invented Hindi medical words ("मांसल निकास", "शरीर में लगाम") and an 82 s wait.

**Diagnosis:** language detection was correct (Hinglish). The translation came out as "What is madhumesh?", keeping the typo'd Hindi word, so keyword search had nothing to match. The 4B model ignored "Latin alphabet" and lacked Hindi medical vocabulary. The 82 s came from sentiment training sharing the CPU.

**Changes:**
1. Translation prompt maps non-English medical words to English terms, fixes typos, keeps the question type (with examples).
2. The model also receives "Meaning in English: …" next to the original question.
3. Hinglish gets an explicit instruction with an example and "do NOT use Devanagari".
4. New rules: answer exactly what was asked (definition first for "what is"); keep English terms in brackets when unsure of a translation.
5. The translated query is stored with each answer (visible under Details).

**Result:** a Hinglish, definition-first, grounded answer in 26 s (first token after 4 s).

---

## Phase 2: improvements (local models only)

Entries are added below as work continues.

### P2.1 Duplicate sources removed (2026-11-23)
**Problem:** the owner saw "What is (are) Causes of Diabetes?" twice in one answer's source cards. Measured on 60 LiveQA questions: two chunks of the same answer took 1% of slots, but **5% of slots (13/240)** went to *different documents with identical text*. MedQuAD copies 304 answer texts verbatim across documents (e.g. NIDDK's diabetes-causes answer exists in 4 documents).
**Change:** `HybridRetriever.search` over-fetches 3× `top_k`, then skips a passage if its (doc_id, title) or its content fingerprint (normalised title + first 300 characters) was already chosen. Keying on (doc, title) rather than doc alone keeps different sections of one help page as separate sources.
**Rejected:** deduplicating at index-build time. It's cleaner, but changing rowids invalidates the embeddings (34 min to redo), and query-time dedup also covers users uploading a document twice.
**Evidence:** new unit test; LiveQA hybrid nDCG@10 0.4505 → 0.4532 (original), dense MRR 0.416 → 0.424; no metric got worse. Every source slot now adds new information for the LLM.

### P2.2 Deterministic emergency safety net (2026-11-23)
**Problem:** emergency handling relied on one line in the medical system prompt. A 4B local model may ignore it, and it didn't apply in General or Research mode at all. A health assistant can't make safety depend on model compliance.
**Change:** `billie/nlp/safety.py` screens every message with explicit patterns for 9 categories (cardiac, breathing, stroke, self-harm, overdose/poisoning, severe bleeding, unconsciousness, seizure, anaphylaxis) in English, Hindi (Devanagari and romanised) and Spanish. A match (1) sends an `alert` in the SSE `meta` event, shown as a red banner above the answer before the model writes anything, with emergency numbers, or crisis helplines (Tele-MANAS 14416, KIRAN, 988) for self-harm; and (2) adds a `SAFETY:` instruction to the final prompt message. The alert is stored with the answer, so reopened chats still show it.
**Design choice:** favour recall over precision. The banner wording ("if this is happening right now…") makes false positives harmless.
**Evidence:** 21 tests (14 languages/phrasings positive, 4 negatives, pipeline integration). On the 104 real LiveQA consumer questions, 5 were flagged: **1 genuine emergency** ("i took 25 iron pills… what do i do") and 4 informational mentions (e.g. "can an abscess cause a heart attack"), where the banner is harmless.

### P2.3 Speed: profiling, answer cache, warm-up and a 2.3 s Windows networking bug (2026-11-23)
**Profiling a real medical answer** (direct Ollama calls with timing stats): the prompt is ~950 tokens, read at ~100 tok/s → 8–11 s; generation at ~9 tok/s → ~11 s per 100 tokens.
- Shrinking `num_ctx` from 8192 to 4096 did **not** speed up prompt reading, and changing it forced a **6 s model reload**. Every call to the same model must use the same `num_ctx` (Billie does).
- An identical prompt is served from Ollama's cache in 0.2 s (vs 23.5 s), but a new question still pays nearly full cost, because the sources vary. Prompt tricks can't change much more on this CPU.

**Changes:**
1. **Answer cache** (`chat/cache.py`, SQLite, LRU 500). The key covers mode + normalised question + provider/model + corpus build stamp + system-prompt hash, so rebuilding an index, switching models or editing prompts invalidates entries. Only first-turn, text-only, LLM-generated, not-ungrounded answers are cached; follow-ups and image questions never are. The lookup happens *before* translation and retrieval. The UI shows a "cached" badge and `/api/status` reports hits.
2. **Warm-up** at server start loads the chat and embedding models in the background, using the same `num_ctx` as real requests.
3. **Bug found while measuring the cache:** cached answers took 2.1 s instead of ~0.1 s. Timing individual calls showed the *first* embedding call after an idle pause took 2,087 ms and later ones 44 ms. Root cause: on Windows, `localhost` resolves to IPv6 `::1` first, but Ollama listens only on IPv4 `127.0.0.1`. Every new connection waited ~2.3 s for the IPv6 attempt to fail (measured: localhost 2,344/2,295/2,323 ms vs 127.0.0.1 331/296/279 ms). It hit chat, embedding and translation calls whenever a keep-alive connection had closed. **Fix:** default to `127.0.0.1`, and `normalize_base_url()` rewrites `localhost` so previously saved settings are fixed too.

**Evidence:** 5 cache tests plus a URL test (109 total). Live: cached repeat 2.1 s → **~0.09 s**; fresh medical answer **42.6 s → 24.9 s** (first token 12.5 s). The chat model is loaded 7 s after server start, before the first question.

### P2.4 Feedback, regenerate/edit, and structured logs (2026-11-23)
**Feedback.** 👍/👎 on every answer (`PUT /api/messages/{id}/feedback`, values 1/-1/0; clicking again clears). Stored in a new `messages.feedback` column. Older databases get the column through an automatic `ALTER TABLE` migration, which is tested against a hand-built pre-migration schema. **A 👎 also evicts that answer from the cache** (each answer stores its cache key), so a disliked answer is never replayed. Totals appear in `/api/status`, the Status dialog and Prometheus (`billie_answer_feedback_total`).
**Regenerate and edit question.** `regenerate: true` on `/api/chat` removes the last question/answer pair (`pop_last_turn`) and answers again, bypassing the cache lookup and overwriting the cached entry. Editing a question uses the same path with new text. In the UI, Regenerate appears on the latest answer and Edit on the latest question.
**Structured logs.** JSON-lines logging (`api/logging_setup.py`). Each request gets an `X-Request-ID` (incoming IDs are accepted only if they match `[A-Za-z0-9._-]{1,64}`, otherwise replaced). Every chat turn logs one line: mode, language, grounded, abstained, cached, model, number of citations, alert, per-stage timings. **No user text is logged** (privacy, asserted in a test).
*Subtlety:* a streamed response body runs after the middleware has returned, so the request-ID context variable is already reset. The chat endpoint captures the ID before streaming and logs the turn itself.
**Evidence:** backend 114 tests (feedback, eviction, regenerate, edit, migration, request-ID logging); frontend 13 tests (regenerate replaces rather than appends; feedback toggling sends 1 then 0).

### P2.5 `.env` modernised (2026-11-23)
The owner's `.env` still used pre-overhaul names (`CHATBOT_API_KEY`, `OPENAI_*`, `VECTOR_STORE_BACKEND`), none of which the new code reads. Kept as `.env.legacy` (git-ignored); a fresh `.env` was created from `.env.example` (development mode, local models, no keys).

### P2.6 Stop button doesn't stop the local model: RESOLVED (2026-11-24)
**Problem (measured):** after the browser disconnects mid-answer, a tiny 0.4 s Ollama request waited **21–36 s**. The abandoned generation keeps running and blocks the CPU and the next question.
**Diagnosis so far:**
- Ollama itself stops correctly: closing an httpx stream after 10 tokens → the next request took **0.4 s**. So the leak is in Billie's layer.
- Hypothesis: on disconnect, Starlette cancels the response but leaves the *sync* generator (pipeline → provider → httpx stream) suspended and never closed, so the connection to Ollama stays open. A standalone check confirmed `iterate_in_threadpool` + `aclose()` does **not** close the inner generator (`closed: False`).
- Added `stream_and_close()` in `api/app.py` (closes the generator in a shielded `finally`) plus a unit test that passes. **But the live server still shows 35.7 s**, and the log line "stream closed early" never appears. Either the `finally` doesn't run under uvicorn on Windows, or `http.disconnect` isn't delivered while the worker thread is blocked.
**Resolution (2026-11-24): the real root cause was the middleware, not the generator.**
A minimal experiment isolated it: a bare FastAPI streaming endpoint noticed a client disconnect in **0.1 s** and ran its cleanup. Adding a single `@app.middleware("http")` (Starlette's `BaseHTTPMiddleware`) meant **nothing ran on the server within 8 s**. `BaseHTTPMiddleware` relays the response body through its own internal stream, so `http.disconnect` never reaches the `StreamingResponse`. Billie had three such middlewares (metrics, security, request ID). The same experiment with a pure ASGI middleware cleaned up in 0.1 s.
**Fix:** rewrote all three as pure ASGI middleware (`api/middleware.py`: they wrap `send`, pass `receive` through and add headers on `http.response.start`). `stream_and_close()` stays, so the generator chain is closed and the httpx stream to Ollama shuts down.
**Lesson:** a passing unit test of the cleanup code didn't prove the real server would *call* it. The minimal reproduction plus a live measurement found the actual cause.
**Evidence:** live test (disconnect after 5 tokens, then time a tiny Ollama request): **35.7 s → 0.6 s**, and `data/api.log` shows "stream closed early (client disconnected); generation stopped". New regression test fails if any `BaseHTTPMiddleware` is registered. 116 tests pass.

### P2.7 Incremental embedding and end-to-end tests (2026-11-24)
**Incremental embedding.** `embed_index` now keeps vectors that already exist for the same model and embeds only new rows, so growing the arXiv dense subset from 20k to 100k papers no longer re-embeds the first 20k (~40 min saved). A different model recomputes everything. Two new tests.
**Browser end-to-end tests.** Playwright (`frontend/e2e/billie.spec.ts`) drives the installed Microsoft Edge (`channel: "msedge"`, no browser download) against the real stack: home page and modes; a streamed, grounded, cited answer with a working citation chip and deep-link URL; the emergency banner plus the **Stop** button; the Settings and Documents dialogs. They're excluded from Vitest and CI, because they need the local models (`npm run e2e` with the app running).

### P2.8 Answer-quality evaluation (2026-11-24)
`backend/evals/answer_quality.py` runs the full live pipeline on the 39 LiveQA questions and measures outcomes, citation support (an LLM judge per cited sentence; self-judging bias noted), **uncited factual sentences** (added after the smoke run showed the model answering from memory), agreement with the NLM reference answers against a top-passage baseline, and latency. It uses a temporary conversation DB and no cache.
**Smoke-run findings that shaped the prompt:** (a) "Noonan syndrome + polycystic kidneys": an honest refusal, but it then summarised the sources *without citations*; (b) "VDRL positive": retrieval missed, and the model explained syphilis testing *from memory*. Correct, but it breaks the sources-only rule.
**Baseline results (39 questions, prompt v1):** grounded 71.8%, abstained 28.2%, ungrounded 0%; relevant source retrieved 59%; cited-sentence support 85.7% (53 supported / 14 partial / 3 unsupported of 70); **77% of answers contained uncited factual sentences (60 sentences)**; similarity to NLM reference answers 0.67 vs 0.59 for the top retrieved passage (the answer was better in 79.5% of questions); p50 latency 25.3 s.
**Judge spot-check (by hand, 8 random cited claims with their evidence):** agreed on 7/8. The miss was lenient: an emergency-advice sentence (from the safety-net instruction, not from the vasculitis passage it cited) was rated PARTIAL where UNSUPPORTED is correct. So the support rate is slightly optimistic; reported with that caveat.
**Prompt v2 (evaluated after the baseline):** every factual sentence must carry a citation or be left out; partial answers give the covered part with citations and state what's missing; no explaining from memory when the documents don't cover a topic.

**Prompt v2 result: a measured negative, reverted.** Same 39 questions, same pipeline. Before comparing, the uncited-sentence filter was fixed: v2's "say what the documents don't cover" rule produced meta-sentences ("The sources do not provide…", "no source confirms…") that the first filter wrongly counted as claims. Both runs were re-scored from their saved answers.

| Metric | v1 (baseline) | v2 (stricter citation rules) |
|---|---|---|
| Answers with uncited factual sentences | 74.4% | 92.3% |
| Uncited factual sentences | 57 | 73 |
| Cited sentences | 70 | 59 |
| Cited-sentence support (judge) | 85.7% | 85.6% |
| Grounded / abstained | 71.8% / 28.2% | 76.9% / 23.1% |
| Similarity to NLM answers | 0.67 | 0.67 |

Firmer wording ("every factual sentence must cite…, otherwise leave it out") made the 4B model cite *less*, not more. **Decision: keep v1.** Lesson: prompt changes on small local models must be measured; intuition about "stronger instructions" was wrong here. The remaining gap (~45% of factual sentences uncited) points to a structural fix rather than more prompt wording, e.g. automatic citation repair that matches uncited sentences to sources by embedding similarity and marks unmatched ones as unverified.

### P2.9 Citation repair (2026-11-24)
**Problem:** ~45% of factual sentences were uncited (P2.8), and stronger prompt wording made it worse.
**Change:** `chat/citations.py` runs after generation. It splits the answer into sentences and lines (Markdown preserved), finds factual uncited sentences (refusals, meta statements about sources and advice are excluded), and compares each with the 4 shown sources by embedding cosine. Above the threshold, the citation is inserted before the final punctuation and recorded in `auto_citations` (sentence, source, score). Below it, the sentence counts as `unverified_sentences`. The UI shows "N sentences not linked to a source", and Details lists every automatic citation with its similarity. ~0.2 s per answer.
**Threshold calibration (from data, not guessed):** on baseline answers, judge-SUPPORTED sentences paired with their cited passage (72 positive pairs) vs the same sentences paired with uncited retrieved passages (140 pairs). At **0.60**: 89% precision, 54% of true pairs recovered, 4% of non-cited pairs admitted. Chosen for precision; real precision is likely higher, since some "negative" passages also support the claim.
**Bug caught by a test:** a Markdown list item without a full stop merged with the next line ("could not find…"), so the bullet was skipped. Sentences now also split at line breaks.
**End-to-end tests:** all 4 Playwright tests pass in Edge against the live stack (home and modes; a grounded, cited answer with a working citation chip and deep link; emergency banner + Stop; dialogs). One locator bug fixed: "Documents" also matched history items containing that word, so the test now uses `exact: true`.
**Evidence:** 5 new tests (123 backend total). Live evaluation, 39 questions, prompt v1 + repair:

| Metric | Baseline | + citation repair |
|---|---|---|
| Uncited factual sentences | 57 | **46** (−19%) |
| Answers with uncited sentences | 74.4% | 69.2% |
| Cited sentences | 70 | 81 |
| Automatic citations added | – | 8 |
| **Support rate of automatic citations** (judge) | – | **93.8%** (7 supported, 1 partial) |
| Support rate of model-written citations | 85.7% | 82.9% |
| Similarity to NLM answers | 0.67 | 0.675 |

Automatic citations were *more* often supported than the model's own, in line with the calibrated ~89% precision. The sample is small (8). 45 sentences stayed unverified at the conservative threshold, and the UI now says so instead of hiding it. (This run shared the CPU with arXiv embedding, so its latency, p50 47 s, isn't comparable.) An evaluation bug was fixed: auto-citation tagging compared un-normalised text and matched nothing; it's re-scored offline from the saved verdicts.

### P2.10 arXiv dense coverage 20k → 50k, demo GIF, docs refresh (2026-11-24)
- Incremental embedding added 30,000 papers (only new rows; the first 20k reused) in ~1 h. arXiv now: 980,539 papers in BM25, **50,000 with dense vectors**, hybrid search live. Running `--embed-limit 100000` later continues from here.
- README demo GIF (`docs/images/demo.gif`, 2.1 MB) recorded with Playwright in Edge (`frontend/e2e/record-demo.mjs`) and assembled with Pillow (`scripts/make_demo_gif.py`, which drops identical frames and holds the last one). It shows a stroke question: the emergency banner, the streamed cited answer, an entity chip, and a citation opening its source.
- README and ARCHITECTURE updated: Phase 2 highlights, the answer-quality table (v1 / v2 / repair), test counts (123 backend, 13 frontend, 4 e2e), the middleware and 127.0.0.1 lessons.

### P2.11 100-question stress test: results, failures and fixes (2026-11-24)
`backend/evals/stress_test.py` + `evals/datasets/stress_test.json`: 96 cases + 4 cache repeats across General (20), Medical (40), Research (25), follow-ups (8) and images (3, generated test images). Kinds covered: short/long/one-word, messy consumer text, typos, Hindi/Hinglish/Bengali/Spanish/French, emergencies, out-of-scope, drug/dosage, sentiment, prompt injection, gibberish. Each case states the expected behaviour (answer / abstain / alert / any), key facts, forbidden content and reply language. Run on the local model, with a temporary DB and no cache.
**Automatic result: 96/100.** Then **every answer was read by hand**:
- English answers: accurate and cited (diabetic neuropathy recognised from a long description; "slapped cheek rash" → Fifth disease; BRCA1/2, celiac, Huntington's inheritance correct). One slip: a Billie-internals answer claimed the mode is auto-selected (the user picks it).
- Refusals correct: World Cup, smartphones, boiling point, movie, Python code; the drug-synthesis request refused; the insulin-dose question answered "needs personalised advice".
- Research: real recent papers with citations; follow-ups resolved correctly ("which is most recent?").
- Images (local gemma3): all 3 read correctly (metformin label, glucose trend 95→142 vs the 126 cutoff, a .docx upload error).
- **Weakness: non-English output of the local 4B model** (Spanish "seda" for "sed", invented Hindi words, nonsense Hinglish glosses, garbled Bengali). Groq's 27B model produced fluent Hindi/Hinglish on the same questions, so non-English quality depends on the model.

**Latency (local qwen3:4b, 96 fresh answers):** mean 26.1 s, median 23.6 s, p90 40.7 s. General 14.4 s, Medical 27.9 s, Research 33.1 s (lists of papers), non-English 38.9 s, images 79 s, refusals 12.9 s, **cached 0.1 s**. Stage means: **generation 23.3 s (the bottleneck, ~9 tok/s)**, first token 8.4 s, translation 1.4 s, citation repair 0.7 s, retrieval 0.25 s, vision 51 s. Answer length vs time correlation 0.46.

**Failures and fixes:**
1. **Self-harm message missed** ("…thinking about **ending** my life", "don't want to live") and **poisoning missed** ("swallowed a whole bottle of pills"). Patterns extended (ending/taking my life, don't want to live, better off dead, swallowed/took + bottle/handful + pills). New tests; still exactly 5/104 flags on LiveQA (no new false positives; "I took my pills this morning" doesn't trigger). Note: even without the banner, the model had replied safely.
2. **Answered from memory with no evidence** ("capital of Australia" → "Canberra"). Now: no retrieved documents + not small talk + no emergency → a fixed "not found" reply in 6 languages (`chat/not_found.py`); the model isn't called. Greetings still go to the model; emergencies always do, so the safety reply is never a canned "not found".
3. **Spanish question answered in English.** Root cause: py3langid gave Spanish only 0.57 because probability leaked to Aragonese/Extremaduran, below the 0.75 threshold. Fix: restrict the classifier to Billie's supported languages + `¿`/`¡` → Spanish. 0 of 104 English LiveQA questions misread afterwards.
4. **Needless empathy openers** on neutral questions (typo'd TB question, French asthma). The sentiment probe (trained on tweets) reads disease words as weakly negative (0.38–0.45) vs real distress 0.49–0.73. Rule: negative only if probability ≥ 0.6 or an explicit emotion word is present.
Re-runs of the affected cases: all pass. Results: `docs/evaluation/stress_test/` (the full run is the main report; see its README).

### P2.12 Cloud models with local fallback: Groq for text, Gemini for images (2026-11-24)
**Decision (owner):** Groq for the deployed/demo version; the local model for development and testing.
- **Groq** (`qwen/qwen3.8-27b`): the same pipeline answered in 0.7–6 s vs ~25 s locally, with clearly better Hindi/Hinglish. Free limits seen in the console: 30 req/min, 1k req/day, **8k tokens/min, 200k tokens/day** for this model. At ~1.2–1.8k tokens per question that's **~4–6 questions/minute and ~110–160/day**.
- **`FallbackProvider`** (`llm/fallback.py`): tries the cloud; if it fails before producing text (error, timeout, no internet, 429), the local `qwen3:4b-instruct` answers. A 429 starts a 60 s cool-down; a mid-answer failure isn't silently restarted. The answer's details record which model answered and why. Answers produced by the fallback aren't cached under the cloud model's key.
- **Images with a text-only cloud model:** vision always used to require an Ollama chat provider, so with Groq active images were silently ignored. Now vision is independent of the chat provider.
- **Gemini for vision:** the key works; `gemini-2.5-*` is closed to new users and `gemini-3.8-flash` returned 503 "high demand", while **`gemini-3.5-flash-lite` read images correctly**. Configured as cloud vision with a 30 s timeout and a local `gemma3` fallback. **Image questions: ~80 s → 3–7 s** end to end.
- **Evaluations always use the local model** (`evals/common.py`; `--use-app-model` to opt in), so test suites never burn the free quota.
- API keys live only in the git-ignored `data/llm_settings.json`, masked in every response. The owner was advised to rotate both keys, since they were pasted in chat.
- Small fix: the model wrote "[image]" as a citation; the prompt now says to refer to "the image" and cite only numbered sources.

### P2.13 Speed work without touching answer quality (2026-11-24)
**Owner constraint:** no shorter answers, no fewer sources. Only changes that keep outputs identical. Rejected for that reason: shorter answers, 3 instead of 4 passages, skipping translation, a fuzzy "similar question" cache, a smaller or more heavily quantized local model.
**Applied (results identical, only timing changes):**
1. **Citation-repair source embeddings computed while the model writes** (the sources are known before generation). Groq: repair 0.43 s → **0.07 s**; local: 0.9–1.5 s → **~0.1 s**.
2. **BM25 and dense search run in parallel** (SQLite releases the GIL; dense is an HTTP call), with the same fusion order. Search mean 0.46 → 0.33 s on Groq runs.
3. **Query-embedding LRU cache.** Sentiment and retrieval embed the same question; it's now computed once (repeat: 25–33 ms → 0.1 ms).
**Measured on the live Groq setup (8 questions, all modes, paced for the rate limit):** mean **2.29 s → 1.33 s**; research 4.0 → 2.0 s and 7.7 → 2.9 s. Part of the first-token/generation drop is Groq network variance, so only the repair and search savings (~0.5 s, more for research) are attributed to the change. All answers stayed grounded with full citations.
**Local Ollama runtime settings, benchmarked and NOT changed:** num_thread 6/8/16 and num_batch 1024 vs default on a real 950-token prompt (prompt cache defeated). Generation stayed at 9.2–9.4 tok/s (16 threads was slower: 7.9); no setting beat the default. Reason: decoding is **memory-bandwidth bound**. Each token reads the ~2.5 GB model (≈ 9 tok/s × 2.5 GB ≈ 23 GB/s, typical laptop DDR4). Only a smaller model, heavier quantization or a GPU would help, and the first two cost quality.
**Checked, not a regression:** one arXiv search took 2.75 s in an eval run. Repeated direct measurement showed 0.44–0.9 s warm; the outlier was a cold start (loading the 61 MB vector matrix). Remaining arXiv cost is FTS5 BM25 over 980k papers (0.4–0.8 s); reducing it would mean scoring fewer candidates, which risks recall, so it's left as is.

### P2.14 Published to GitHub (2026-11-24)
- Pushed to `Vishaljhaaa/Billie` `main` as 4 commits (8730da4), authored by the owner, as a normal fast-forward on top of the existing history.
- A secret scan of the published tree found no real keys. The only matches are fake test keys.
- Kept out of the repository: `.env*`, `data/` (API keys, indexes, sessions), datasets, virtualenvs and `node_modules`.

### P2.15 Public demo mode and a deployable bundle (2026-11-24)
**Goal:** let an interviewer try Billie from a link, without the free Groq quota being drained by anyone who finds it.

**Problems found while planning:**
1. **Everyone shared one chat history.** `/api/conversations` returned every conversation, so a public demo would show visitors each other's chats.
2. **Production startup required a chat API key,** which a public demo can't hand out.
3. **Unlimited questions.** The free quota is ~1k requests and 200k tokens per day, i.e. ~110–160 questions. One curious visitor or a bot could use it all.
4. **Status checks cost quota.** `/api/status` runs a cloud health check (`GET /models`) on every page load, and Groq counts those calls.
5. **The sentiment model was missing from installs.** `pip install ./backend` (the Docker path) left out `nlp/models/*.npz`, because setuptools ignores non-Python files without `package-data`. A Docker image would have silently fallen back to the word list.
6. **The full arXiv index is 2.6 GB** (980k papers), but only 50k papers have embeddings.
7. **Framing.** The CSP `frame-ancestors 'none'` and `X-Frame-Options: DENY` would block the Space page, which shows the app in an iframe.

**Changes:**
- **`BILLIE_DEMO=1`** (`api/demo.py`):
  - Each browser sends a random `X-Visitor-Id`. Conversations get an `owner` column (migrated in place) holding a SHA-256 of that id.
  - Every conversation query (list, get, rename, delete, feedback, and continuing a chat) is scoped to the owner. Someone else's chat behaves exactly like one that doesn't exist (404); continuing it starts a new chat instead.
  - Locally, `owner=None` keeps the single-user behaviour unchanged.
- **Daily quotas:** 25 questions per visitor IP and 150 in total per UTC day, with a friendly 429 message.
  - The personal cap uses the IP, not the visitor id, because clearing browser storage would reset the id.
  - Answers served from the cache are refunded, since they cost no model call.
  - Counters live in memory; a restart only errs towards availability.
- **Production rules:** a demo needs only `BILLIE_ADMIN_KEY` (settings and the knowledge base stay admin-only). A private production deployment still needs both keys.
- **Status cost:** in demo mode the LLM health result is cached for 60 s and cleared when settings change.
- **New settings via environment variables:** `BILLIE_LOCAL_FALLBACK`, `BILLIE_VISION_CLOUD`, `BILLIE_VISION_CLOUD_MODEL`, and `BILLIE_FRAME_ANCESTORS` (CSP allow-list; drops `X-Frame-Options`, which can't express one).
- **`python -m billie.ingest bundle --out DIR`** (`retrieval/index.subset_index`) copies MedQuAD whole and only the embedded arXiv papers.
  - Rowids are remapped to the vector matrix order, and the vectors are copied unchanged.
  - Result: 2.6 GB → **150 MB + 147 MB of vectors**. Spot checks returned the same hybrid results on both corpora.
- **Packaging:** added `[tool.setuptools.package-data]` so the sentiment head ships with the package.
- **UI:** a "Public demo · N of 25 questions left today" banner, refreshed after every answer.
- **Hugging Face files** in `deploy/huggingface/`:
  - A three-stage Dockerfile: the React build; Ollama 0.40.17 with `embeddinggemma` pulled at build time (GPU libraries deleted); and a Python 3.12 slim image serving on 7860 as uid 1000.
  - `start.sh` starts Ollama, waits until it answers, then runs `uvicorn --proxy-headers` so rate limits see the real visitor IP.
  - The Space card README, and `stage.py`, which builds the upload folder and refuses to stage anything named `.env*` or `llm_settings.json`.
- **Bug fix:** the stress-test p90 used `ordered[int(0.9*(n-1))]`, which returns the minimum for n=2 (p90 < median in an earlier summary). It now uses nearest rank, `ordered[ceil(0.9n)-1]`.

**Tests:** 10 new tests in `tests/test_demo.py`, covering quota maths and the daily reset, visitor isolation for every route (including continuing someone else's chat), feedback scoping, the 429 and cache refund, admin-only settings and documents in demo mode, unscoped local mode, frame headers, production rules, environment defaults, and `subset_index` with a dense-only search. **151 backend tests pass**; ruff and mypy are clean.

**Blocker: Hugging Face now charges for Docker Spaces.** Creating the Space returned `402 Payment Required`: "hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription". Only static Spaces stay free, and a static page can't keep API keys secret. Nothing was created and no secrets were sent. The bundle and Dockerfile are ready either for HF PRO or for another Docker host with ≥2 GB RAM. Hosts with 512 MB free tiers (Render, Koyeb) would need the embeddings moved to an API and the corpora re-embedded, a quality-relevant change that would have to be measured first.

**Frontend test runner false alarm:** Vitest reported "failed to find the current suite" for every file, even with the changes stashed. Root cause: the shell's working directory used a lowercase drive letter (`c:\...`) while Vitest resolved `C:\...`, so Windows loaded the `vitest` module twice under two paths. Run from an uppercase `C:` path, all **13/13 tests pass**, and `tsc` and `vite build` are clean. It was environmental, not a code regression; noted for anyone running the tests from Git Bash on Windows.

### P2.16 Full stress test on the cloud setup (2026-11-24)
**Why:** the README only had cloud numbers from 8 questions, and the owner wanted the default (Groq + Gemini) setup measured as thoroughly as the local model. This deliberately used ~100 Groq requests, paced at 12 s for the 8k tokens/min limit. No demo depended on the quota that day.

**Results (96 fresh questions + 4 cache repeats):**
- Latency: mean **2.0 s**, median 1.7 s, p90 3.1 s, max 13.4 s (one general question).
- By mode: General 2.3 s, Medical 1.8 s, Research 2.2 s. Non-English 3.0 s, images 4.4 s, emergencies 1.2 s, cached 0.0 s.
- Stages: first token 0.96 s, generation 1.33 s, retrieval 0.16 s, citation repair 0.02 s, translation 1.17 s, vision 2.97 s.
- Hindi, Hinglish, Bengali, Spanish and French answers read fluently and were cited. One word slip: Hinglish "siyahi" (ink) used for wheezing.

**First score 95/100; all 5 "failures" were correct refusals.** The cases were coding, smartphone, meth synthesis, boiling point and movie recommendation. The 27B model declines and then cites what the sources *do* cover ("the sources only describe its health effects [1]"). The verdict logic treated any valid citation as *grounded*, and the eval failed abstain cases that had citations.

**Fix:** `opens_with_refusal()` in `chat/pipeline.py`. An answer whose first sentence declines counts as **abstained** even with citations; English plus Spanish, French, German and Hinglish openers are covered. A unit test uses real answers from this run, and negatives check that later "cannot" sentences aren't misread. The eval now accepts abstained answers with citations.

**Re-scoring:** applying the same rule to the saved results moved 7 answers to *abstained*: the 5 above, plus two honest "the sources don't say" answers (ibuprofen side effects, inventor of the transformer). The result is **100/100**. The local full run wasn't re-scored; it stays the record from before its own fixes.

**Bug found on the way:** the stress-test p90 formula (see P2.15).

### P2.17 Live public demo on Azure Container Apps (2026-11-25)
**Hosting decision:** Hugging Face Docker Spaces need PRO (P2.15). Free alternatives were checked against Billie's ~2 GB memory need:
- Render's free tier has 512 MB and sleeps when idle.
- Oracle Always Free needs a card, and idle servers can be reclaimed.
- Cloud Run needs a card.
- The Heroku student credit only covers 512 MB.
- DigitalOcean's student credit ended in July 2026.

**Chosen: Azure for Students** ($100 credit per year, no card, so no real-money risk) with **Container Apps**, which scales to zero.

**Constraints found on the student subscription:**
1. A policy allows only 5 regions (UAE North, India South Central, East Asia, Indonesia Central, Malaysia West). Container Apps isn't offered in India South Central, so the app runs in **UAE North**, the closest allowed region.
2. **ACR Tasks are blocked** (`TasksOperationsNotAllowed`), so Azure can't build the image itself, and there's no Docker on the laptop.
   - Workaround: a manual GitHub Actions workflow (`build-image.yml`) builds `deploy/huggingface/Dockerfile` and pushes it to the registry.
   - The ~400 MB index bundle is attached to a release.
   - Only a repository owner can store Actions secrets on a personal-account repo, so the build runs from the owner's fork (`Max0786234/Billie`). The fork holds the registry password secret and the index release. Code still lives in `Vishaljhaaa/Billie`.

**Resources** (resource group `billie-rg`):
- Container Registry, Basic tier: ~$5/month, the only fixed cost
- Container Apps environment with log storage off
- Container App `billie`: 2 vCPU / 4 GiB, min 0 / max 1 replica, external ingress on port 7860
- Secrets: Groq key, Gemini key, admin key
- A $10/month budget alert (80% actual, 100% forecast)

The keys were read from local files and passed straight to Azure; they were never printed, committed or logged.

**Verification on the live URL:**
- `/api/status`: Groq ready, `embeddinggemma` available, all three corpora on hybrid search (knowledge 22, MedQuAD 27,582 passages / 16,407 vectors, arXiv 50,000 / 50,000). Production and demo mode on.
- Questions, all grounded and cited:
  - medical (English): 1.7 s
  - Hinglish "madhumeh hota kya hai": 2.3 s, answered in Hinglish
  - research: 2.6 s, citing 5 papers
  - general: 1.0 s
- Image questions: 24 s the first time (vision 21.9 s, a Gemini latency outlier), then **5.8 s** (vision 3.7 s).
- Privacy and access: a second visitor id sees 0 of the first visitor's 5 chats; changing settings without the admin key returns 401; the quota counter decremented as expected.

### P2.18 Showing users what Billie can answer (2026-11-25)
**Problem (owner feedback):** a friend tried the live demo, asked general questions, got "not in the documents" several times and didn't know what Billie was for. A retrieval system is only as useful as its coverage, but the start screen said "Medical" or "Research" without saying what was inside. Every out-of-scope question also cost a model call.

**Changes:**
1. **Coverage card per mode.** `GET /api/overview` (`corpora/overview.py`) is computed from the live index, so the live site (50k papers) and a full local index (980k) each show their own numbers:
   - **Medical:** 16,407 answers on 5,126 topics, the NIH sources, and question types (What is, Symptoms, Treatment, Inheritance…)
   - **Research:** paper count, years and the biggest categories
   - **General:** the documents
   - every mode: "good for" and "not covered" lists

   Scanning 980k arXiv rows takes ~18 s, so the stats are computed in the background at startup and cached in `<index>.overview.json` per index build. Until then the endpoint serves the basic counts.
2. **Examples that work.** Each mode's examples were taken from stress-test questions that passed on the default setup, including Hindi, Hinglish and Spanish ones.
3. **Suggestions as you type** (`GET /api/suggest`). A prefix search on index *titles* (FTS5 `title : ("word" "prefix"*)`): MedQuAD titles are real questions, and arXiv titles are papers.
   - For MedQuAD, titles are cleaned up ("What is (are) X ?" → "What is X?", "What are the treatments for X" → "How is X treated?", doubled "??" removed) and short titles are shown first. Long ones are page-title quirks such as "Who is at risk for Am I at Risk for Type 2 Diabetes? …??".
   - Measured: 3–20 ms on MedQuAD, ~0.1–0.8 s over 980k papers locally (debounced in the UI), and fast on the 50k live index.
4. **Dead ends turned into next steps** (only when the answer abstained or nothing was found; ~60 ms):
   - **Better mode:** the question's embedding is compared with the best passage in each corpus.
   - **Calibration:** in-scope questions scored ≥ 0.48 against their corpus, off-topic ones ≤ 0.37 everywhere (15 sample questions). So a mode is suggested only if it scores ≥ 0.45 *and* beats the current one by ≥ 0.08. One click re-asks there.
   - **Closest topics in this mode:** only passages with cosine ≥ 0.45 to the question, and none at all when another mode fits better.
     - The first browser run showed asbestos and osteoporosis offered for a BM25 question.
     - Checked on 8 dead-end cases: off-topic passages scored ≤ 0.38 (now hidden); near misses 0.39–0.47.
     - One known false match remains: "VDRL test" matches Von Hippel-Lindau disease at 0.47.
   - A refusal no longer shows "1 sentence not linked to a source": a refusal makes no claims to verify.
5. The header taglines no longer hard-code corpus sizes; they said "~980,000 papers" on a site that has 50,000.

**Tests:** 8 new backend tests (`tests/test_guidance.py`) cover the overview, its disk cache, prefix suggestions, title cleaning, related topics on abstained answers, no guidance on grounded answers, the better-mode rule with the calibrated values, and the endpoints. There are also 2 new frontend tests for the guide.
- Backend **160 passed**, frontend **15 passed**; ruff, mypy and tsc clean.
- Browser check in Edge (`e2e/guide-check.mjs`):
  - the coverage cards and suggestions render
  - a BM25 question asked in Medical mode declines and suggests General mode
  - one click switches and gives a grounded answer in 0.6 s

API checks on fresh questions in Medical mode:
- the BM25 question suggests General and shows no noise topics
- "best smartphone camera" gets only an honest decline
- "ibuprofen side effects" offers osteoarthritis treatment and immune thrombocytopenia

**Deployed** as image `b2171d5`, built by the fork's workflow; `az containerapp update` switched the app to it. On the live site: the suggestions return MedQuAD questions, the overview shows the live index (50,000 papers from 2025–2026), and the BM25 question asked in Medical mode suggests General mode with no noise topics.

### P2.19 Visitors' own documents: private and temporary (2026-11-25)
**Request (owner):** let demo visitors use their own documents, deleted when they leave so storage isn't wasted. Before this change, uploads on the demo were admin-only, because they went into the shared knowledge base, so visitors saw "missing admin key".

**Design** (`private_docs.py`, enabled only in demo mode):
- **Searched with Billie's docs.** Each visitor's files get a separate small index (BM25 + embeddings), stored under their hashed visitor id. In General mode, Billie searches it together with Billie's docs.
  - Both result lists carry RRF scores, which depend only on ranks, so they merge directly. On a tie the visitor's file comes first.
  - Sources are labelled "Your document · name". Ids are namespaced as `private:…`.
- **"When they leave" means 1 hour after last use.** A browser can't reliably report a closed tab, so an inactivity timer replaces it.
  - Uploading, listing and asking all count as use.
  - A background sweep runs every `min(300 s, ttl/4)`.
  - Removing a file, or any restart, also deletes it; the folder is wiped on start because the store is temporary by design.
  - The lifetime is configurable: `BILLIE_PRIVATE_DOCS_TTL_MINUTES`.
- **Never cached.** When a visitor has files, General-mode answers skip the shared answer cache, both lookup and store. Otherwise a cached answer built from one visitor's file could be served to another. Tested.
- **Limits:**
  - 3 files per visitor, 5 MB per file, and 150 passages per file (~25 dense pages, measured in P2.20; this bounds embedding time on 2 CPUs)
  - 300 MB total on disk and 100 visitors holding files at once
  - the existing rate limit and daily question caps

  Files are validated by type, encoding and PDF signature. Unreadable files, scanned PDFs (no text layer), oversized files and wrong types get clear messages, and a rejected file leaves nothing behind.
- **UI:** the Documents dialog becomes "Your private documents" on the demo.
  - It shows a privacy note ("passages are sent to the AI model"), the limits, passage counts and the time until deletion.
  - General mode's start screen shows a "Have your own document?" link.
  - Admin uploads to the shared knowledge base are still reachable from the dialog.

**Problems found by testing with real files** (demo mode on port 8010, local qwen3:4b-instruct, so no cloud quota was used):
1. **An oversized file took 88 s to be rejected.** All its passages were embedded before the length check.
   - Fix: parse the new file on its own and check its length and text before any embedding.
   - Now rejected in 0.03 s.
2. **Every upload re-embedded all of the visitor's files.**
   - Fix: embeddings cached per file (`<file>.vectors.npz`, tagged with the model) and the vector matrix reassembled in rowid order. Adding a second file embedded only it: 0.26 s for a Markdown file. Unit-tested with a counting fake embedder.
3. **"What is the daily meal allowance?" wasn't found** although the PDF says 1,800 rupees. Knowledge chunks are 900 characters, but the prompt shows at most 650 per source, so the end of the travel section was cut off.
   - Fix: private documents use 600-character chunks.
   - After the fix the answer was found and cited.

**Results on real documents** (a 3-page PDF handbook and Markdown trip notes, all on the local model):
- Correct cited answers (cited as "Your document"):
  - 24 days of leave, up to 5 carried over: 20.7 s
  - the meal allowance (1,800 rupees): 6.7 s
  - the remote-work approver (Priya Raman): 8.4 s
  - password length (14 characters): 3.9 s
  - the Leh flight (06:15): 10 s
  - the Hinglish question "remote work hafte mein kitne din…" got "3 din [1]": 17 s
- Billie's own docs still answer next to the private files ("How does hybrid search work?" cited `how_billie_works.md`).
- **Isolation:**
  - Visitor B's listing is empty, B gets a 404 deleting A's file, and B asking the same question gets an honest "not found" with no private sources.
  - Nothing was cached.

**Tests:** 12 new backend tests (`tests/test_private_docs.py`) cover:
- indexing and search, labels and namespacing
- expiry with a fake clock, and the wipe on start
- every limit and bad-file case, and the visitor and storage caps
- invalid owners
- API isolation, the cache exclusion, shared docs still answering, 404 outside the demo, clean 400s
- the per-file embedding cache

**Deployed** as image `0bfacfa`. Checked on the live site:
- the handbook PDF uploaded in 1.2 s
- a fake PDF was rejected
- another visitor's listing was empty
- "daily meal allowance?" was answered on Groq in 0.8 s: "1,800 rupees [1]", cited as "Your document", not cached

The test file was removed afterwards.

### P2.20 Evaluating answers over visitors' own documents (2026-11-25)
**Question (owner):** are answers correct for the documents a user provides, and does a local test say anything about Groq + Gemini?

**Method** (`evals/private_docs_eval.py`; data in `evals/datasets/private_docs/`). Two PDFs are uploaded as one demo visitor:
- a generated **22-page city annual report** (`make_city_report.py`). Its known facts sit in prose and tables, among plausible filler that shares their vocabulary, so search has real distractors.
- the real **20-page project report**, which has real tables.

**30 questions:**

| Kind | Count |
|---|---|
| Plain facts | 11 |
| Table values | 8 |
| Combining two figures (e.g. a difference or a sum) | 4 |
| Lists | 2 |
| Summary | 1 |
| Not answered by the documents | 4 |

**Two scores:**
- **Retrieval:** did a source contain the answer, or the figures it's computed from? This is model-independent, because the search code and embedding model are identical locally and live.
- **Answer correctness:** checked automatically against expected facts. For unanswerable questions, the answer must decline without inventing a figure.

**Why a local run says something about Groq:** search and embeddings are identical, so retrieval results transfer exactly. Writing quality is a lower bound, so a 10-question Groq spot-check (~10 requests) confirms it.

**First run: 25/30 (retrieval 22/26). Failures traced to real bugs:**
1. **Uploading the project report crashed** with `UNIQUE constraint failed: passages.id`.
   - Passage ids were (file, section title, chunk number), so a repeated heading, or `# …` lines inside code samples (which parse as Markdown headings), produced the same id twice. This also affected shared admin uploads.
   - Because the rebuild ran outside the error handling, the visitor got a 500 and the file stayed half-added.
   - Fixes: the section's position is now part of the id, and an indexing failure now rolls back the upload with a clear message. Regression test added.
2. **A PDF collapsed into a single source.**
   - Search drops repeated chunks of the same (document, title). That rule is right for MedQuAD, where one answer is split into chunks, but PDF text has no headings, so every chunk of a 137-passage report shared one title and only one survived.
   - This caused "Which department had the largest budget?" → "Public transport" (Billie saw half the table), and two missed facts in the project report. Long sections in the shared knowledge base were affected the same way.
   - Fix: `collapse_chunks` applies only to MedQuAD and arXiv; documents keep every distinct chunk, and identical text is still removed. Regression test added.
3. **Three scoring mistakes in the evaluation itself** (fixed, never by changing answers):
   - a refusal flagged for "2023." echoed from the question;
   - a computed sum checked against the sources (which can only contain its parts);
   - years and citation markers like "[1]" counted as invented figures.

**Final results (`docs/evaluation/private_docs/`; the first run is kept as `*_before_fixes.json`):**

| | Local qwen3:4b-instruct (30 questions) | Groq qwen3.8-27b (10 hardest) |
|---|---|---|
| Correct | **28/30** | **9/10** |
| Right passage among the sources | 25/26 | 8/9 |
| Facts, tables, combined figures, lists, unanswerable | 10/11, 8/8, 4/4, 2/2, 4/4 | 2/2, 3/3, 2/2, 1/1, 1/1 |
| Whole-document summary | 0/1 | 0/1 |
| Mean time per answer | 9.2 s | 0.9 s |

**What's left, and why:**
- **"Summarise the report" fails on both models.** Billie answers from the 4 most relevant passages, and a 22-page report can't be summarised from 4 passages, so it picks up two real achievements plus generic text. This is a limit of question-driven retrieval, not of the model. A real fix would need a separate summarise path (e.g. sampling passages across the whole document, or summarising section by section). That's possible, but slow on a CPU.
- **One local miss:** asked for the Stop-button timings, the 4B model found "20–35 s before" but missed "35.7 s → 0.6 s" in its sources. Groq answered it correctly (35 s → 0.6 s).
- **Page limit corrected:** dense pages produce ~6 passages each. The 22-page report made 137 passages, close to the 150 limit, so the upload message now says "about 25 pages of dense text" instead of 40.
- **Upload time locally:** 20.7 s for the 137-passage PDF and 15.5 s for 88 passages (embedding on the laptop CPU). The live site uses the same CPU-only embedding on 2 vCPUs, so expect a similar or longer wait.

---

## Remaining backlog (updated 2026-11-24, end of P2.16)
Open items:
1. **Redeploying after code changes:**
   1. `gh repo sync Max0786234/Billie`
   2. `gh workflow run build-image.yml -R Max0786234/Billie`
   3. `az containerapp update -g billie-rg -n billie --image billiemaxacr.azurecr.io/billie:<sha>`
2. Grow arXiv dense coverage further: `python -m billie.ingest arxiv --embed-only --embed-limit 100000` (~1 h per 30k on this CPU).
3. Refresh the README screenshots (they predate feedback buttons and citation repair).
4. Retrieval for misspelled or abbreviated medical terms (e.g. "VDRL"): measure an English query-expansion step before adopting it.
5. Owner decision: delete the `repo/` and `zip_extract/` scratch folders.
