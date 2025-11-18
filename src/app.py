import pandas as pd
import numpy as np
import nltk
import joblib
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
import re
# Making the NN model
from keras.utils import to_categorical
from keras.models import Sequential 
from keras.layers import Dense, LSTM, Dropout
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('wordnet')

data = pd.read_csv("./data/spam-ham.csv")
data ["msg_len"] = data["Message"].apply(len)
stop_words = stopwords.words("english")

# print(data.head())
# print(data.info())
# print(data.shape)
#print(stop_words)

spam_data = data[data["Category"] == "spam"]
ham_data = data[data["Category"] == "ham"]

def get_numbers(s):
    return len(re.findall(f"\d{5,}", s))

def get_num_urls(s):
    return len(re.findall(f"(http|https|www)\:\/\/[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(\/\S*)?", s))

def clean_message(msg):
    msg = msg.lower()
    msg = re.sub("[^a-zA-Z]", " ", msg)
    msg = nltk.word_tokenize(msg)
    lemma = WordNetLemmatizer()
    msg = [lemma.lemmatize(word) for word in msg if word not in stop_words and len(word) >1]
    msg = " ".join(msg)
    return msg

data["number_count"] = data["Message"].apply(get_numbers)
data["url_count"] = data["Message"].apply(get_num_urls)

data = data[data["msg_len"] < 183]

data["clean_msg"] = data["Message"].apply(clean_message)
data["is_spam"] = data["Category"].apply(lambda x: 1 if x == "spam" else 0)
data.drop(labels=["Message", "Category"], inplace=True, axis=1)


tfidf = TfidfVectorizer(max_df=.8, max_features=300)
tfidf_mat = tfidf.fit_transform(data.iloc[:,3].values)
Y = data["is_spam"].values

X = np.hstack((tfidf_mat.toarray(), data.iloc[:, 0:3].values))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.25, stratify=Y)
X_train_nn = X_train.reshape(-1, X_train.shape[1], 1)
X_test_nn = X_test.reshape(-1, X_test.shape[1], 1)
y_train_nn = to_categorical(y_train)
y_test_nn = to_categorical(y_test)

model = Sequential()

# Units - outputs dimension 
model.add(LSTM(units=50, activation="relu", return_sequences=True))
model.add(Dropout(rate=0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))

model.add(Dense(units=2))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train_nn, y_train_nn, epochs=7, batch_size=32, verbose=1, validation_data=(X_test_nn, y_test_nn))

model.save("lstm_spam_model.h5")        # save model
joblib.dump(tfidf, "tfidf_vectorizer.pkl")  # save TF-IDF