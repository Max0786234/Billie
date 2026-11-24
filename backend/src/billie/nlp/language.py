"""Language identification.

1. **Script rules** - text mostly in Devanagari, Bengali, Arabic, Cyrillic, CJK, ...
   scripts identifies the language family with near certainty.
2. **Romanised Hindi** ("Hinglish") - very common in India, and statistical
   classifiers trained on standard text misread it. A cue-word check catches it.
3. **py3langid** (langid.py, Lui & Baldwin 2012; 97 languages, naive Bayes over byte
   n-grams) for Latin-script text.
4. Very short or low-confidence Latin text defaults to English, because "hi",
   "thanks" or "BM25?" carry no reliable language signal.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "bn": "Bengali", "es": "Spanish", "fr": "French", "de": "German",
    "pt": "Portuguese", "it": "Italian", "nl": "Dutch", "ru": "Russian", "ar": "Arabic", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ta": "Tamil", "te": "Telugu", "mr": "Marathi", "gu": "Gujarati",
    "pa": "Punjabi", "ur": "Urdu", "tr": "Turkish", "id": "Indonesian", "vi": "Vietnamese", "pl": "Polish",
    "uk": "Ukrainian", "fa": "Persian", "th": "Thai", "sw": "Swahili", "ml": "Malayalam", "kn": "Kannada",
}  # fmt: skip

# (unicode name prefix, language) - first matching script with a majority wins.
_SCRIPTS = [
    ("DEVANAGARI", "hi"), ("BENGALI", "bn"), ("ARABIC", "ar"), ("CYRILLIC", "ru"), ("HANGUL", "ko"),
    ("HIRAGANA", "ja"), ("KATAKANA", "ja"), ("CJK", "zh"), ("TAMIL", "ta"), ("TELUGU", "te"),
    ("GUJARATI", "gu"), ("GURMUKHI", "pa"), ("THAI", "th"), ("MALAYALAM", "ml"), ("KANNADA", "kn"),
]  # fmt: skip

_HINGLISH_CUES = frozenset(
    "kya hai hain mujhe mera meri mere kaise kaisa kyun kyon nahi nahin karu karoon karna kar raha rahi "
    "ho hota hoti tha thi aap tum mai main hum bata batao samjhao kuch bahut accha acha theek dard ilaj "
    "dawai dawa bimari kab kahan kitna kitne".split()
)
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass(frozen=True)
class LanguageGuess:
    code: str
    name: str
    confidence: float
    method: str  # "script", "hinglish", "langid", "default"
    reply_language: str  # what to put in "Reply in ..."


def detect_language(text: str) -> LanguageGuess:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return _guess("en", 0.5, "default")

    script = _dominant_script(letters)
    if script:
        lang = script
        # Japanese text mixes kanji (CJK) with kana; any kana means Japanese.
        if lang == "zh" and any("HIRAGANA" in _name(c) or "KATAKANA" in _name(c) for c in letters):
            lang = "ja"
        return _guess(lang, 0.99, "script")

    words = [w.casefold() for w in _WORD.findall(text)]
    cues = sum(w in _HINGLISH_CUES for w in words)
    if cues >= 2 and cues / max(len(words), 1) >= 0.25:
        return LanguageGuess("hi", "Hindi", 0.8, "hinglish", "Hinglish (Hindi written in the Latin alphabet)")

    if "¿" in text or "¡" in text:  # inverted punctuation exists only in Spanish
        return _guess("es", 0.95, "script")
    if len(words) < 3:
        return _guess("en", 0.6, "default")
    code, probability = _identifier().classify(text)
    if code not in LANGUAGE_NAMES or probability < 0.75:
        return _guess("en", 0.5, "default")
    return _guess(code, float(probability), "langid")


def _guess(code: str, confidence: float, method: str) -> LanguageGuess:
    name = LANGUAGE_NAMES.get(code, code)
    return LanguageGuess(code, name, confidence, method, name)


def _dominant_script(letters: list[str]) -> str | None:
    counts: dict[str, int] = {}
    for ch in letters:
        if ch.isascii():
            continue
        name = _name(ch)
        for prefix, lang in _SCRIPTS:
            if name.startswith(prefix):
                counts[lang] = counts.get(lang, 0) + 1
                break
    if not counts:
        return None
    lang, count = max(counts.items(), key=lambda item: item[1])
    return lang if count / len(letters) >= 0.4 else None


def _name(ch: str) -> str:
    return unicodedata.name(ch, "")


@lru_cache(maxsize=1)
def _identifier():
    from py3langid.langid import MODEL_FILE, LanguageIdentifier

    identifier = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
    # Only consider languages Billie names. Otherwise probability leaks to close relatives nobody
    # types (Aragonese, Extremaduran for Spanish), a clear Spanish question scores 0.57 and gets
    # rejected as low-confidence (found by the stress test).
    identifier.set_languages([code for code in LANGUAGE_NAMES if code in identifier.nb_classes])
    return identifier
