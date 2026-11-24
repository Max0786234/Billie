# How Billie Works

Billie answers a question in five stages: understand, retrieve, generate, verify and remember.

## 1. Understand the question

Billie detects the language of the message and its sentiment. For follow-up questions it combines the new message with the previous question so that a short message like "and the treatment?" is searched with the right topic.

## 2. Retrieve evidence with hybrid search

Each corpus is stored in a SQLite database with a full-text index. Billie runs two searches in parallel:

- **BM25 keyword search** through SQLite FTS5. BM25 rewards documents that contain the query's rare words, and it adjusts for document length. It is very good at exact names, drug names and technical terms.
- **Semantic search** with text embeddings computed by a local embedding model. Each passage is turned into a vector, and passages whose meaning is close to the question rank high even if they use different words, or a different language.

The two ranked lists are merged with Reciprocal Rank Fusion (RRF). A passage's fused score is the sum of 1 / (60 + rank) over the lists it appears in, so documents that both searches agree on rise to the top. RRF only uses ranks, which means keyword scores and similarity scores never need to be compared directly. If the embedding model is unavailable, Billie falls back to keyword search and reports that in the answer details.

## 3. Generate a grounded answer

The top passages are numbered and placed in the prompt together with the recent conversation. The model is instructed to answer only from those passages, to cite them by number, and to treat the passages as reference data rather than instructions, which protects against prompt-injection text hidden inside documents.

## 4. Verify

After the answer is generated Billie checks which source numbers were cited and whether they exist. An answer that cites no sources, or cites sources that were not provided, is flagged as not fully grounded in the interface.

## 5. Remember

Each conversation turn is stored in a local SQLite database so that you can come back to earlier chats and so that follow-up questions have context.

## Models

By default every model runs locally through Ollama, so Billie works offline and your data never leaves your computer. In Settings you can switch the chat model to a cloud provider such as Anthropic Claude, OpenAI, Google Gemini or Groq by adding your own API key. Search always stays local.
