"""Train and evaluate the multilingual sentiment head.

Data: cardiffnlp/tweet_sentiment_multilingual (Barbieri et al., 2022) - 8 languages,
3 classes, official train/validation/test splits. Downloaded once through the public
Hugging Face datasets-server API and cached under evals/datasets/sentiment/.

    python -m evals.train_sentiment                 # train on train+validation, report on test
    python -m evals.train_sentiment --per-lang 600  # faster: subsample training data

Writes the head to billie/nlp/models/ and metrics to docs/evaluation/sentiment/.
"""

from __future__ import annotations

import argparse
import json
import time

import httpx
import numpy as np

from billie.nlp.lexicon_sentiment import lexicon_sentiment
from billie.nlp.sentiment import LABELS, SentimentHead, model_path, train_softmax
from billie.paths import EVAL_DATASETS_DIR, REPO_ROOT
from billie.retrieval.embeddings import OllamaEmbedder

DATASET = "cardiffnlp/tweet_sentiment_multilingual"
LANGUAGES = ["english", "spanish", "hindi", "french", "german", "italian", "portuguese", "arabic"]
CACHE = EVAL_DATASETS_DIR / "sentiment"
RESULTS = REPO_ROOT / "docs" / "evaluation" / "sentiment"


def fetch(config: str, split: str) -> list[dict]:
    path = CACHE / f"{config}_{split}.jsonl"
    if path.exists():
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows: list[dict] = []
    with httpx.Client(timeout=60) as client:
        offset = 0
        while True:
            payload = _get_page(client, config, split, offset)
            rows.extend({"text": r["row"]["text"], "label": r["row"]["label"]} for r in payload["rows"])
            offset += 100
            if offset >= payload["num_rows_total"]:
                break
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return rows


def _get_page(client: httpx.Client, config: str, split: str, offset: int, attempts: int = 8) -> dict:
    """One page of rows, backing off politely when the public API rate-limits us (HTTP 429)."""
    params = {"dataset": DATASET, "config": config, "split": split, "offset": offset, "length": 100}
    delay = 5.0
    for _ in range(attempts):
        response = client.get("https://datasets-server.huggingface.co/rows", params=params)
        if response.status_code in (429, 502, 503):
            wait = float(response.headers.get("retry-after", delay))
            print(f"  rate limited, waiting {wait:.0f}s")
            time.sleep(wait)
            delay = min(delay * 2, 120)
            continue
        response.raise_for_status()
        time.sleep(0.5)  # be gentle with a free public API
        return response.json()
    raise RuntimeError(f"gave up fetching {config}/{split} at offset {offset}")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: int = 3) -> float:
    scores = []
    for c in range(classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="embeddinggemma")
    parser.add_argument("--per-lang", type=int, default=None, help="cap training rows per language")
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--fetch-only", action="store_true", help="download and cache the dataset, then exit")
    args = parser.parse_args()

    if args.fetch_only:
        for lang in LANGUAGES:
            for split in ("train", "validation", "test"):
                print(f"{lang}/{split}: {len(fetch(lang, split))} rows")
        return 0

    embedder = OllamaEmbedder(args.model, batch_size=64)
    if not embedder.available():
        print(f"Embedding model '{args.model}' is not available in Ollama. Run: ollama pull {args.model}")
        return 1

    rng = np.random.default_rng(0)
    train_texts: list[str] = []
    train_labels: list[int] = []
    test: dict[str, tuple[list[str], np.ndarray]] = {}
    for lang in LANGUAGES:
        rows = fetch(lang, "train") + fetch(lang, "validation")
        if args.per_lang and len(rows) > args.per_lang:
            rows = [rows[i] for i in rng.choice(len(rows), args.per_lang, replace=False)]
        train_texts += [r["text"] for r in rows]
        train_labels += [r["label"] for r in rows]
        test_rows = fetch(lang, "test")
        test[lang] = ([r["text"] for r in test_rows], np.array([r["label"] for r in test_rows]))
        print(f"{lang:11} train={len(rows):5}  test={len(test_rows)}")

    started = time.perf_counter()
    x_train = embedder.embed_queries(train_texts)
    print(f"embedded {len(train_texts)} training texts in {time.perf_counter() - started:.0f}s")
    weights, bias = train_softmax(x_train, np.array(train_labels), l2=args.l2)
    head = SentimentHead(weights, bias, args.model)

    lexicon_map = {"negative": 0, "neutral": 1, "positive": 2}
    report: dict[str, dict] = {}
    all_true, all_pred = [], []
    for lang, (texts, y_true) in test.items():
        probs = head.predict_proba(embedder.embed_queries(texts))
        y_pred = probs.argmax(axis=1)
        lex_pred = np.array([lexicon_map[lexicon_sentiment(t).label] for t in texts])
        report[lang] = {
            "n": len(texts),
            "accuracy": round(float(np.mean(y_pred == y_true)), 4),
            "macro_f1": round(macro_f1(y_true, y_pred), 4),
            "lexicon_baseline_accuracy": round(float(np.mean(lex_pred == y_true)), 4),
            "lexicon_baseline_macro_f1": round(macro_f1(y_true, lex_pred), 4),
        }
        all_true.append(y_true)
        all_pred.append(y_pred)
        print(f"{lang:11} acc={report[lang]['accuracy']:.3f}  F1={report[lang]['macro_f1']:.3f}  "
              f"(lexicon acc={report[lang]['lexicon_baseline_accuracy']:.3f} F1={report[lang]['lexicon_baseline_macro_f1']:.3f})")  # fmt: skip
    y_true_all, y_pred_all = np.concatenate(all_true), np.concatenate(all_pred)
    overall = {
        "accuracy": round(float(np.mean(y_pred_all == y_true_all)), 4),
        "macro_f1": round(macro_f1(y_true_all, y_pred_all), 4),
    }
    print(f"overall     acc={overall['accuracy']:.3f}  F1={overall['macro_f1']:.3f}")

    # Second, out-of-domain check: hand-written customer-support messages from the original project.
    support = json.loads((EVAL_DATASETS_DIR / "sentiment_evaluation.json").read_text(encoding="utf-8"))
    support_true = np.array([LABELS.index(r["label"]) for r in support])
    support_pred = head.predict_proba(embedder.embed_queries([r["text"] for r in support])).argmax(axis=1)
    support_lex = np.array([lexicon_map[lexicon_sentiment(r["text"]).label] for r in support])
    support_report = {
        "n": len(support),
        "accuracy": round(float(np.mean(support_pred == support_true)), 4),
        "macro_f1": round(macro_f1(support_true, support_pred), 4),
        "lexicon_baseline_accuracy": round(float(np.mean(support_lex == support_true)), 4),
    }
    print(
        f"support msgs acc={support_report['accuracy']:.3f}  (lexicon {support_report['lexicon_baseline_accuracy']:.3f})"
    )

    head.save(model_path(args.model))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"results-{args.model}.json").write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "embedding_model": args.model,
                "classifier": "multinomial logistic regression (numpy), class-weighted, L2",
                "labels": list(LABELS),
                "train_rows": len(train_texts),
                "l2": args.l2,
                "overall_test": overall,
                "per_language_test": report,
                "support_messages": support_report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved head to {model_path(args.model)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
