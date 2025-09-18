# 基于字符的编码

import tensorflow as tf
from keras import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

def readTxtFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as file:
        data = file.read()
    return data



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






#
# if __name__ == "__main__":
#     data = readTxtFile("./XW_ab.txt")
#     data = data.split()
#     print(len(set(data)))