import pandas as pd
from nltk.corpus import stopwords


data = pd.read_csv("./data/spam-ham.csv")
stop_words = stopwords.words("english")
# print(data.head())
# print(data.info())
# print(data.shape)

