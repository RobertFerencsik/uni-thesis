# spam_lstm/config.py

MAX_WORDS = 10000
MAX_LEN = 120
EMBED_DIM = 64
LSTM_UNITS = 64
BATCH_SIZE = 64
EPOCHS = 6
TEST_SIZE = 0.25
RANDOM_STATE = 42

MODEL_PATH = "models/lstm_spam_model.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"

