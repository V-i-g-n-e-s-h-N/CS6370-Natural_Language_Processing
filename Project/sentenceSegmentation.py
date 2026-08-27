import re
from nltk.tokenize import sent_tokenize


class SentenceSegmentation:

    def naive(self, text):
        if not text or not text.strip():
            return []
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in parts if s.strip()]

    def punkt(self, text):
        if not text or not text.strip():
            return []
        return sent_tokenize(text)
