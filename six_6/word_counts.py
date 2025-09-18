from read_json import read_json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

from collections import OrderedDict

# 嵌入维度最好是词汇表大小的4次方根
sentences, labels, urls = read_json('./JSON-sarcasm/Sarcasm_Headlines_Dataset_v2.json')

training_size = 23000
vocab_size = 2000 # 10000
embedding_size = 7# 16

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

training_sentences = np.array(training_sentences)
testing_sentences = np.array(testing_sentences)
training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="oov_token")
tokenizer.fit_on_texts(sentences)
wc = tokenizer.word_counts

new_list = (OrderedDict(sorted(wc.items(), key=lambda x: x[1], reverse=True)))
print(new_list)