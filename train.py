# train.py

from spam_lstm.data.dataset import SpamDataset
from spam_lstm.data.preprocess import Preprocessor
from spam_lstm.trainer import Trainer

print("Loading dataset...")
ds = SpamDataset("./db/train/spam_ham.csv")
df = ds.get_dataframe()

print("Preprocessing...")
df["msg_len"] = df["Message"].apply(len)
df["number_count"] = df["Message"].apply(Preprocessor.count_numbers)
df["url_count"] = df["Message"].apply(Preprocessor.count_urls)
df["clean_msg"] = df["Message"].apply(Preprocessor.clean_message)
df["is_spam"] = df["Category"].apply(lambda x: 1 if x == "spam" else 0)

trainer = Trainer(df)
Xtext_train, Xtext_test, Xmeta_train, Xmeta_test, y_train, y_test = trainer.prepare()

print("Training model...")
trainer.train(Xtext_train, Xtext_test, Xmeta_train, Xmeta_test, y_train, y_test)

print("Done!")
