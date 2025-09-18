# 双向LSTM文本生成，复合预测以生成文本

import numpy as np
from six_6 import read_json
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer

def predict_class(model, x):
    prediction = model.predict(x, verbose=0)
    return np.argmax(prediction, axis=-1)

data =  ("In the town of Athy one Jeremy Lanigan\n"
         "Battered away til he hadnt a pound.\n"
         "His father died and made him a man again\n"
         "Left him a farm and ten acres of ground.\n"
         "He gave a grand party for friends and relations\n"
         "Who didnt forget him when come to the wall,\n"
         "And if youll but listen Ill make your eyes glisten\n"
         "Of the rows and the ructions of Lanigan’s Ball.\n"
         "Myself to be sure got free invitation,\n"
         "For all the nice girls and boys I might ask,\n"
         "And just in a minute both friends and relations\n"
         "Were dancing round merry as bees round a cask.\n"
         "Judy ODaly, that nice little milliner,\n"
         "She tipped me a wink for to give her a call,\n"
         "And I soon arrived with Peggy McGilligan\n"
         "Just in time for Lanigans Ball.\n")

tokenizer = Tokenizer()
corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
totalWords = len(tokenizer.word_index) + 1

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
# 独热编码，把标签变成独热编码，有k个可能的了别，向量长度就是k，属于哪个类别，就把对应的位置变为1，其他的为0
ys = tf.keras.utils.to_categorical(labels, num_classes=totalWords)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(totalWords, 8))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(max_sequence_len - 1)))
model.add(tf.keras.layers.Dense(totalWords, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(xs, ys, epochs=1500, verbose=1)


seed_text = "sweet jeremy saw dublin"
next_words = 10

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

    prediction = predict_class(model, token_list)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == prediction:
            output_word = word
            break
    seed_text = seed_text + " " + output_word

print(seed_text)