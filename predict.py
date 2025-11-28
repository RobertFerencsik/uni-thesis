# predict.py

import json
import numpy as np
import sys
from spam_lstm.data.preprocess import Preprocessor
from spam_lstm.model.lstm_model import LSTMSpamClassifier
from spam_lstm.model.tokenizer_wrapper import TextTokenizer

def predict_from_json(json_path):
    """
    Predicts spam/ham from a JSON file containing a message.

    Args:
        json_path (str): Path to the JSON file.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    msg = data["message"]

    print("Preprocessing message...")
    cleaner = Preprocessor()
    clean_msg = cleaner.clean_message(msg)
    msg_len = len(msg)
    num_count = cleaner.count_numbers(msg)
    url_count = cleaner.count_urls(msg)

    print("Loading model and tokenizer...")
    # Load model and tokenizer
    model_wrapper = LSTMSpamClassifier()
    model_wrapper.load()
    tokenizer_wrapper = TextTokenizer()
    tokenizer_wrapper.load()

    # Prepare inputs for the model
    X_text = tokenizer_wrapper.transform([clean_msg])
    X_meta = np.array([[msg_len, num_count, url_count]])

    print("Making prediction...")
    # Predict
    pred = model_wrapper.model.predict([X_text, X_meta])[0]
    label = "SPAM" if np.argmax(pred) == 1 else "HAM"

    print(f"\nMessage: '{msg}'")
    print(f"Prediction: {label} (spam score = {pred[1]:.4f})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_json>")
        sys.exit(1)
    
    predict_from_json(sys.argv[1])