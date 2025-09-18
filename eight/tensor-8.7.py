# 基于字符的编码

import tensorflow as tf
from keras import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

def readTxtFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as file:
        data = file.read()
    return data

def predict_class(model, x):
    prediction = model.predict(x, verbose=0)
    return np.argmax(prediction, axis=-1)

data = readTxtFile("./XW_ab.txt")
corpus = data.split()
tokenizer = Tokenizer(char_level=True, oov_token='oov', filters='', lower=False)  # 基于字符的编码 不过滤任何字符
tokenizer.fit_on_texts([data])
ids = tokenizer.texts_to_sequences([data])[0]
totalChars = len(tokenizer.word_index) + 1

block_size = 100
xs, label = [], []

for i in range(0, len(ids) - block_size):
    x = ids[i:i + block_size]
    y = ids[i + block_size]
    xs.append(x)
    label.append(y)

xs, ys = np.array(xs), tf.keras.utils.to_categorical(np.array(label))

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(totalChars, 3))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(totalChars - 1, return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(totalChars - 1)))
model.add(tf.keras.layers.Dense(totalChars, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(xs, ys, epochs=50, verbose=1, batch_size=512)


seed_text = "You know nothing, John Snow."
next_words = 75


for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=block_size - 1, padding='pre')

    prediction = predict_class(model, token_list)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == prediction:
            output_word = word
            break
    seed_text = seed_text + " " + output_word
print(seed_text)


#
# if __name__ == "__main__":
#     data = readTxtFile("./XW_ab.txt")
#     data = data.split()
#     print(len(set(data)))