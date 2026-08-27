import re
from nltk.tokenize import word_tokenize


class Tokenization:

    def naive(self, text):
        tokenized_text = []
        for sent in text:
            tokenized_text.append(re.findall(r"\b\w+\b", sent))
        return tokenized_text

    def pennTreeBank(self, text):
        return [word_tokenize(sent) for sent in text]
