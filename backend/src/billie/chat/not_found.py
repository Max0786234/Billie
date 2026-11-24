"""Deterministic reply when retrieval finds nothing and the message is not small talk.

Found by the stress test: asked "What is the capital of Australia?" with no matching
documents, the local model answered "Canberra" from memory despite the instruction not to.
When there is no evidence, the safe answer must not depend on the model obeying, so the
"not found" reply is a fixed, translated template. Greetings and thanks still go to the
model so they get a natural reply.
"""

from __future__ import annotations

import re

_SMALL_TALK = re.compile(
    r"^\W*(hi+|hello|hey|hiya|yo|good (morning|afternoon|evening|night)|thanks?( you| a lot| so much)?|thank u|ty|"
    r"bye|goodbye|see you|ok(ay)?|cool|great|nice|awesome|who are you|what can you do|help|"
    r"namaste|namaskar|dhanyavaad|dhanyavad|shukriya|hola|gracias|adi[oó]s|bonjour|merci|salut|"
    r"नमस्ते|नमस्कार|धन्यवाद|शुक्रिया|নমস্কার|ধন্যবাদ)\b[\W\w]{0,30}$",
    re.IGNORECASE,
)

CORPUS_NAMES = {
    "general": "the knowledge base",
    "medical": "the medical knowledge base (MedQuAD)",
    "research": "the arXiv paper index",
}

_TEMPLATES = {
    "en": "I couldn't find anything about this in {corpus}, so I can't answer it reliably. "
    "Try rephrasing your question, or switch to a different mode.",
    "hinglish": "Mujhe {corpus} mein iske baare mein kuch nahi mila, isliye main iska sahi jawab nahi de sakta. "
    "Apna sawaal dusre shabdon mein poochiye ya koi aur mode chuniye.",
    "hi": "मुझे {corpus} में इसके बारे में कुछ नहीं मिला, इसलिए मैं इसका भरोसेमंद जवाब नहीं दे सकता। "
    "कृपया सवाल दूसरे शब्दों में पूछें या कोई दूसरा मोड चुनें।",
    "bn": "{corpus}-এ এ বিষয়ে কিছু পাইনি, তাই নির্ভরযোগ্য উত্তর দিতে পারছি না। প্রশ্নটি অন্যভাবে জিজ্ঞাসা করুন বা অন্য মোড বেছে নিন।",
    "es": "No encontré nada sobre esto en {corpus}, así que no puedo responder de forma fiable. "
    "Intenta reformular la pregunta o cambia de modo.",
    "fr": "Je n'ai rien trouvé à ce sujet dans {corpus}, je ne peux donc pas répondre de façon fiable. "
    "Essayez de reformuler la question ou de changer de mode.",
}


def is_small_talk(text: str) -> bool:
    return bool(_SMALL_TALK.match(text.strip()))


def not_found_message(mode: str, language: str, reply_language: str) -> str:
    key = "hinglish" if reply_language.startswith("Hinglish") else language
    template = _TEMPLATES.get(key, _TEMPLATES["en"])
    return template.format(corpus=CORPUS_NAMES.get(mode, "the documents"))
