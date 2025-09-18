# 8.4 是扩展数据集
# 8.5 改变模型框架，采用多个堆叠lstm
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer


def predict_class(model, x):
    prediction = model.predict(x, verbose=0)
    return np.argmax(prediction, axis=-1)

data =  ("In the town of Athy one Jeremy Lanigan "
         "Battered away til he hadnt a pound. "
         "His father died and made him a man again "
         "Left him a farm and ten acres of ground."
         "He gave a grand party for friends and relations "
         "Who didnt forget him when come to the wall,"
         "And if youll but listen Ill make your eyes glisten "
         "Of the rows and the ructions of Lanigan’s Ball. "
         "Myself to be sure got free invitation, "
         "For all the nice girls and boys I might ask, "
         "And just in a minute both friends and relations "
         "Were dancing round merry as bees round a cask. "
         "Judy ODaly, that nice little milliner, "
         "She tipped me a wink for to give her a call, "
         "And I soon arrived with Peggy McGilligan "
         "Just in time for Lanigans Ball. ")
window_size = 10
sentences_local = []
alltext = []

tokenizer = Tokenizer()
corpus = data.lower()
tokenizer.fit_on_texts([corpus])
totalWords = len(tokenizer.word_index) + 1


words = corpus.split(" ")
range_size = len(words) - window_size

for i in range(0, range_size):
    thissentence = ""
    for w in range(0, window_size - 1):
        word = words[i + w]
        thissentence = thissentence + " " + word
    sentences_local.append(thissentence.strip())

input_sequences = []
for line in sentences_local:

    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=window_size, padding='pre'))

xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
# 独热编码，把标签变成独热编码，有k个可能的了别，向量长度就是k，属于哪个类别，就把对应的位置变为1，其他的为0
ys = tf.keras.utils.to_categorical(labels, num_classes=totalWords)



model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(totalWords, 8))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(window_size - 1, return_sequences=True)))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(window_size - 1)))
model.add(tf.keras.layers.Dense(totalWords, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(xs, ys, epochs=1500, verbose=1)


seed_text = "sweet jeremy saw dublin"
next_words = 10

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=window_size - 1, padding='pre')

    prediction = predict_class(model, token_list)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == prediction:
            output_word = word
            break
    seed_text = seed_text + " " + output_word

print(seed_text)
