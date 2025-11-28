# spam_lstm/model/lstm_model.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout, Bidirectional, Concatenate
)

from spam_lstm.config import MAX_WORDS, MAX_LEN, EMBED_DIM, LSTM_UNITS, MODEL_PATH

class LSTMSpamClassifier:
    def __init__(self):
        self.model = None

    def build(self):
        # text input
        text_input = Input(shape=(MAX_LEN,))
        x = Embedding(MAX_WORDS, EMBED_DIM)(text_input)
        x = Bidirectional(LSTM(LSTM_UNITS))(x)
        x = Dropout(0.5)(x)

        # metadata input
        meta_input = Input(shape=(3,))

        # combined model
        combined = Concatenate()([x, meta_input])
        combined = Dense(32, activation="relu")(combined)
        combined = Dropout(0.3)(combined)

        output = Dense(2, activation="softmax")(combined)

        self.model = Model(inputs=[text_input, meta_input], outputs=output)
        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return self.model

    def save(self):
        self.model.save(MODEL_PATH)

    def load(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)
