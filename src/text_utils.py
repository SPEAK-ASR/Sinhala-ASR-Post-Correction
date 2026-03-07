import re
import unicodedata
from typing import List

# Sinhala Unicode ranges
SINHALA_VOWEL_SIGNS = "ාිීුූෙේෛොෝෞංඃ"
SINHALA_DIGITS = {
    "0": "බිංදුව", "1": "එක", "2": "දෙක", "3": "තුන",
    "4": "හතර", "5": "පහ", "6": "හය", "7": "හත",
    "8": "අට", "9": "නව",
}
SINHALA_TENS = {
    "10": "දහය", "20": "විස්ස", "30": "තිහ", "40": "හතළිහ",
    "50": "පනහ", "60": "හැට", "70": "හැත්තෑව", "80": "අසූව",
    "90": "අනූව",
}


def nfc_normalize(text: str) -> str:
    """NFC normalization — critical for Sinhala Unicode consistency."""
    return unicodedata.normalize("NFC", text)


def is_sinhala(text: str) -> bool:
    """Return True if the text contains at least one Sinhala character."""
    return any("\u0D80" <= ch <= "\u0DFF" for ch in text)


def clean_text(text: str) -> str:
    """NFC-normalize, collapse whitespace, and strip the text."""
    text = nfc_normalize(str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentence_split(text: str, max_len: int = 200) -> List[str]:
    """Split long text into sentence-length chunks suitable for training."""
    sentences = re.split(r"(?<=[.!?।෴])\s+", text)
    chunks: List[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) < max_len:
            current += " " + sent
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sent
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 10]
