import random
import re
from typing import List, Optional

from src.text_utils import clean_text


class SinhalaASRNoiseAugmenter:
    """
    Simulates common Whisper ASR errors on clean Sinhala text.
    Used only for DatasetEntry items that have apply_noise=True.
    """

    VOWEL_SIGNS = list("ාිීුූෙේෛොෝෞංඃ")
    SIMILAR_CHARS = {
        "ක": ["ඛ", "ග"], "ට": ["ඩ", "ත"], "ප": ["බ", "ඵ"],
        "ස": ["ශ", "ෂ"], "ල": ["ළ"], "ණ": ["න"], "ඩ": ["ද"],
    }
    SPOKEN_NUMBERS = {
        "1": "එක", "2": "දෙක", "3": "තුන", "4": "හතර", "5": "පහ",
        "6": "හය", "7": "හත", "8": "අට", "9": "නව", "10": "දහය",
        "11": "එකොළහ", "12": "දොළහ", "14": "දාහතර", "15": "පහළොව",
        "20": "විස්ස", "22": "දෙවිසි", "100": "සිය", "1000": "දාහ",
    }
    PUNCTUATION = r"[.!?,;:–—\"'()\[\]]"

    def __init__(self, noise_prob: float = 0.3) -> None:
        self.noise_prob = noise_prob

    def drop_vowel_signs(self, text: str) -> str:
        return "".join(
            c for c in text
            if c not in self.VOWEL_SIGNS or random.random() > self.noise_prob
        )

    def substitute_similar_chars(self, text: str) -> str:
        chars = list(text)
        for i, ch in enumerate(chars):
            if ch in self.SIMILAR_CHARS and random.random() < self.noise_prob * 0.5:
                chars[i] = random.choice(self.SIMILAR_CHARS[ch])
        return "".join(chars)

    def remove_punctuation(self, text: str) -> str:
        return re.sub(self.PUNCTUATION, "", text).strip()

    def numbers_to_spoken(self, text: str) -> str:
        for digit, spoken in sorted(self.SPOKEN_NUMBERS.items(), key=lambda x: -len(x[0])):
            text = re.sub(r"\b" + re.escape(digit) + r"\b", spoken, text)
        return text

    def add_filler_words(self, text: str) -> str:
        fillers = ["ඇ", "හ්ම්", "ම්", "ඔව්"]
        words = text.split()
        if len(words) > 3 and random.random() < self.noise_prob:
            pos = random.randint(1, len(words) - 1)
            words.insert(pos, random.choice(fillers))
        return " ".join(words)

    def apply(self, text: str, error_types: Optional[List[str]] = None) -> str:
        all_types = [
            "drop_vowel_signs",
            "substitute_similar_chars",
            "remove_punctuation",
            "numbers_to_spoken",
            "add_filler_words",
        ]
        if error_types is None:
            selected = ["remove_punctuation"]
            selected += [t for t in all_types[:-1] if random.random() < 0.5]
        else:
            selected = error_types

        noisy = text
        for error_type in selected:
            noisy = getattr(self, error_type)(noisy)
        return clean_text(noisy)
