"""Arabic text normalisation used on both sides (book and transcript) before matching."""
from __future__ import annotations

import re

# Single-codepoint honorifics that a reader pronounces as several words.
_LIGATURES = {
    "ﷺ": " صلى الله عليه وسلم ",   # ﷺ
    "ﷻ": " جل جلاله ",             # ﷻ
    "﵀": " رحمه الله ",
    "﵁": " رضي الله عنه ",
    "﵂": " رضي الله عنها ",
    "﵃": " رضي الله عنهم ",
    "﵄": " رضي الله عنهما ",
    "﵅": " رضي الله عنهن ",
    "﵇": " عليه السلام ",
    "ﷲ": " الله ",
}
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_LETTERS = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي", "ؤ": "و", "ة": "ه",
})
_NOT_ARABIC_LETTER = re.compile(r"[^ء-ي\s]")


def normalize(text: str) -> str:
    for lig, spoken in _LIGATURES.items():
        if lig in text:
            text = text.replace(lig, spoken)
    text = _DIACRITICS.sub("", text)
    text = text.translate(_LETTERS)
    return _NOT_ARABIC_LETTER.sub(" ", text)


def tokens(text: str) -> list[str]:
    return normalize(text).split()
