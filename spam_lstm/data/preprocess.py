# spam_lstm/data/preprocess.py

import re
import nltk
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

stop_words = stopwords.words("english")
lemma = WordNetLemmatizer()

class Preprocessor:

    @staticmethod
    def clean_message(msg: str) -> str:
        msg = msg.lower()
        msg = re.sub("[^a-zA-Z]", " ", msg)
        tokens = nltk.word_tokenize(msg)
        tokens = [
            lemma.lemmatize(w) for w in tokens
            if w not in stop_words and len(w) > 1
        ]
        return " ".join(tokens)

    @staticmethod
    def count_numbers(msg: str) -> int:
        return len(re.findall(r"\d{5,}", msg))

    @staticmethod
    def count_urls(msg: str) -> int:
        return len(re.findall(r"(http|https|www)", msg))
