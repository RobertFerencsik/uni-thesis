# spam_lstm/trainer.py

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from spam_lstm.config import TEST_SIZE, RANDOM_STATE, EPOCHS, BATCH_SIZE
from spam_lstm.model.lstm_model import LSTMSpamClassifier
from spam_lstm.model.tokenizer_wrapper import TextTokenizer

class Trainer:
    def __init__(self, df):
        self.df = df

    def prepare(self):
        texts = self.df["clean_msg"].values
        labels = self.df["is_spam"].values
        meta = self.df[["msg_len", "number_count", "url_count"]].values
        
        tokenizer = TextTokenizer()
        tokenizer.fit(texts)
        X_text = tokenizer.transform(texts)
        y = to_categorical(labels)

        Xtext_train, Xtext_test, Xmeta_train, Xmeta_test, y_train, y_test = \
            train_test_split(X_text, meta, y, test_size=TEST_SIZE, 
                             stratify=labels, random_state=RANDOM_STATE)

        self.tokenizer = tokenizer
        return Xtext_train, Xtext_test, Xmeta_train, Xmeta_test, y_train, y_test

    def train(self, Xtext_train, Xtext_test, Xmeta_train, Xmeta_test, y_train, y_test):
        model = LSTMSpamClassifier()
        model.build()

        history = model.model.fit(
            [Xtext_train, Xmeta_train],
            y_train,
            validation_data=([Xtext_test, Xmeta_test], y_test),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1
        )

        model.save()
        self.tokenizer.save()
        return history
