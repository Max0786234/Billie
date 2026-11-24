"""Deterministic emergency detection: a safety net that never depends on the LLM.

A small local model might ignore "tell the user to call emergency services". A health
assistant must not rely on that, so messages are also screened with explicit patterns
(English, Hindi in Devanagari and Latin script, Spanish). A match:
  * shows an emergency banner in the UI, whatever the model writes, and
  * adds an explicit instruction to the prompt to lead with emergency advice.

Design choice: favour recall over precision. A banner on "what are heart attack
symptoms?" costs nothing; a missed emergency could cost a life. The wording ("if this
is happening now…") keeps false positives harmless.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

EMERGENCY_NUMBERS = "112 (India) · 911 (US) · 999 (UK) · 112 (EU)"

_PATTERNS: dict[str, list[str]] = {
    "self_harm": [
        r"\bsuicid", r"\bkill(ing)? my ?self\b", r"\bend(ing)? (my|it all|my own) life\b", r"\bwant(ed)? to die\b",
        r"\bend(ing)? it all\b", r"\btak(e|ing) my (own )?life\b", r"\b(don'?t|do not|no longer) want to (live|be alive)\b",
        r"\bbetter off dead\b", r"\bno (point|reason) (in )?(living|to live|going on)\b",
        r"\bself[- ]?harm", r"\bhurt(ing)? myself\b", r"\bno reason to live\b", r"\bcut(ting)? myself\b",
        r"aatmahatya", r"khudkushi", r"mar(na|ne) (chahta|chahti)", r"jeena nahi (chahta|chahti)",
        r"आत्महत्या", r"खुदकुशी", r"मरना चाहत", r"जीना नहीं चाहत",
        r"\bsuicidarme\b", r"\bquiero morir\b", r"\bmatarme\b",
    ],
    "cardiac": [
        r"\bchest (pain|pressure|tightness)\b", r"\bpain in (my|the) chest\b", r"\bheart attack\b",
        r"\bcardiac arrest\b", r"seene (me|mein|mai) dard", r"chhati (me|mein) dard",
        r"सीने में दर्द", r"छाती में दर्द", r"दिल का दौरा", r"\bdolor (de|en el) pecho\b", r"\binfarto\b",
    ],
    "breathing": [
        r"\b(can'?t|cannot|unable to|hard to|difficulty|trouble) breath", r"\bnot breathing\b",
        r"\bshort(ness)? of breath\b", r"\bchoking\b", r"saans (nahi|nhi|lene me|lene mein)",
        r"सांस (नहीं|लेने में)", r"साँस (नहीं|लेने में)", r"\bno puedo respirar\b",
    ],
    "stroke": [
        r"\bstroke\b", r"\bface (is )?droop", r"\bslurred speech\b", r"\bsudden (numbness|weakness)\b",
        r"\bone side of (my|his|her|the) (body|face)\b", r"lakwa", r"लकवा", r"\bderrame cerebral\b",
    ],
    "overdose": [
        r"\boverdos", r"\btook too many (pills|tablets)\b", r"\bswallowed (poison|bleach)\b", r"\bpoison(ed|ing)\b",
        r"\b(swallowed|ate|took|taken|drank)\b.{0,30}\b(bottle|pack|packet|strip|handful|lot|many|all)\b.{0,20}"
        r"\b(pills|tablets|medicines?|capsules)\b",
        r"zeher (kha|pi)", r"ज़हर|जहर", r"\bsobredosis\b",
    ],
    "bleeding": [r"\b(heavy|severe|won'?t stop|uncontrolled) bleeding\b", r"\bbleeding (heavily|a lot|won'?t stop)\b"],
    "unconscious": [
        r"\bunconscious\b", r"\bunresponsive\b", r"\bpassed out\b", r"\bnot waking up\b", r"behosh", r"बेहोश",
    ],
    "seizure": [r"\bseizure", r"\bconvuls", r"\bfits?\b.*\b(now|again)\b", r"\bdaura pad", r"मिर्गी का दौरा"],
    "anaphylaxis": [
        r"\banaphyla", r"\bthroat (is )?(swelling|closing)\b", r"\bswollen (tongue|throat)\b",
        r"\bsevere allergic reaction\b",
    ],
}  # fmt: skip

_COMPILED = {category: [re.compile(p, re.IGNORECASE) for p in patterns] for category, patterns in _PATTERNS.items()}

_MESSAGES = {
    "self_harm": (
        "If you are thinking about suicide or hurting yourself, you are not alone and help is available right now. "
        "Please call Tele-MANAS 14416 or KIRAN 1800-599-0019 (India), 988 (US), or your local emergency number."
    ),
    "default": (
        "If you or someone near you is experiencing this right now, call emergency services immediately: "
        f"{EMERGENCY_NUMBERS}. Do not wait for an online answer."
    ),
}


@dataclass(frozen=True)
class EmergencyAlert:
    categories: list[str] = field(default_factory=list)
    message: str = ""

    @property
    def triggered(self) -> bool:
        return bool(self.categories)


def detect_emergency(text: str) -> EmergencyAlert:
    categories = [c for c, patterns in _COMPILED.items() if any(p.search(text) for p in patterns)]
    if not categories:
        return EmergencyAlert()
    message = _MESSAGES["self_harm"] if "self_harm" in categories else _MESSAGES["default"]
    return EmergencyAlert(categories, message)


PROMPT_HINT = (
    "SAFETY: the message may describe a medical emergency or a risk of self-harm. Begin your reply by telling the "
    "user, kindly and clearly, to contact emergency services or a crisis helpline now, then answer briefly."
)
