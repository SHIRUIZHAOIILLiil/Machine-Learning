import numpy as np
from six_6 import read_json
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer


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

# print(input_sequences)

# print(totalWords)