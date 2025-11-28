# spam_lstm/model/tokenizer_wrapper.py

import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from spam_lstm.config import MAX_WORDS, MAX_LEN, TOKENIZER_PATH

class TextTokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<UNK>")

    def fit(self, texts):
        self.tokenizer.fit_on_texts(texts)

    def transform(self, texts):
        seq = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    def save(self):
        joblib.dump(self.tokenizer, TOKENIZER_PATH)

    def load(self):
        self.tokenizer = joblib.load(TOKENIZER_PATH)
