import numpy as np

import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import json

with open('intents.json') as file:
    data =json.load(file)
train_lines=[]
train_labels = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        train_lines.append(pattern)
        train_labels.append(intent['tag'])
        responses.append(intent['responses'])
num = len(train_labels)

label = LabelEncoder()
label.fit(train_labels)
train_labels = label.transform(train_labels)
vocab_size = 1000
embedding_dim= 16

tokenizer = Tokenizer(num_words=vocab_size,oov_token="<OOV>")
tokenizer.fit_on_texts(train_lines)
word_index = tokenizer.word_index
seq = tokenizer.texts_to_sequences(train_lines)
pad_seq = pad_sequences(seq, truncating="post", maxlen = 20)

model = Sequential()
model.add(Embedding (vocab_size , embedding_dim, input_length = 20))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(num , activation = "softmax"))

model.compile(loss ='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(pad_seq, np.array(train_labels), epochs=500)

model.save("chat_model.keras")

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer,handle,protocol= pickle.HIGHEST_PROTOCOL)

with open('label_encoder.pickle','wb') as ecn_file:
    pickle.dump(label, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

