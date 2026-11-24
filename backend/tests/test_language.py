from __future__ import annotations

import pytest

from billie.nlp.language import detect_language


@pytest.mark.parametrize(
    ("text", "code", "method"),
    [
        ("What are the symptoms of glaucoma?", "en", "langid"),
        ("hi", "en", "default"),
        ("What is BM25?", "en", "langid"),
        ("¿Cuáles son los síntomas de la diabetes?", "es", "script"),
        ("¿Cómo se trata la hipertensión?", "es", "script"),  # was misread as English (stress test)
        ("Comment traiter l asthme ?", "fr", "langid"),
        ("Quels sont les traitements du cancer?", "fr", "langid"),
        ("Was sind die Symptome von Asthma?", "de", "langid"),
        ("मधुमेह के लक्षण क्या हैं?", "hi", "script"),
        ("ডায়াবেটিসের লক্ষণ কী?", "bn", "script"),
        ("mujhe sir dard ho raha hai kya karu", "hi", "hinglish"),
        ("糖尿病的症状是什么？", "zh", "script"),
        ("糖尿病の症状は何ですか？", "ja", "script"),
        ("12345 ?!", "en", "default"),
    ],
)
def test_detect_language(text, code, method):
    guess = detect_language(text)
    assert (guess.code, guess.method) == (code, method)


def test_hinglish_reply_language_keeps_latin_script():
    assert "Latin" in detect_language("mujhe bukhar hai kya karu").reply_language
