# 使用正则化降低拟合
# 本质上是在损失函数里额外加一个惩罚项，限制参数不能无限大
# L1 正则化 lasso 最小绝对收缩和选择算子：损失函数中加入权重绝对值之和，鼓励参数稀疏化变为0，不重要的权重就会被压到0
# 做特征选择
# L2 正则化 ridge 岭回归，鼓励参数变小，防止数值过大，更稳定

from read_json import read_json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences, labels, urls = read_json('./JSON-sarcasm/Sarcasm_Headlines_Dataset_v2.json')

training_size = 23000
vocab_size = 2000 # 10000
embedding_size = 7 # 16

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

training_sentences = np.array(training_sentences)
testing_sentences = np.array(testing_sentences)
training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="oov_token")
tokenizer.fit_on_texts(training_sentences)
# 为词汇表中每一个词添加一个16维的向量，一个词对应16个值
# 每个词经过嵌入层都会有一个16纬向量
# tf.keras.layers.Embedding(vocab_size, embedding_size)

# 首先把训练集的所有句子变成数字序列
X_train = tokenizer.texts_to_sequences(training_sentences)
X_test = tokenizer.texts_to_sequences(testing_sentences)

# 然后把数字序列填充为相同长度的矩阵
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post', truncating='post', maxlen=100)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, padding='post', truncating='post', maxlen=100)

model = tf.keras.Sequential([
    # 先从权重矩阵取出每个词的向量，如果embedding_size是16，那么每个词就是16维向量
    # 再把每个词的16维向量组成一个词矩阵

    tf.keras.layers.Embedding(vocab_size, embedding_size),
    # 降维，假设一句话是四个词，每个词是3维向量，那么把每个词的同一向量加起来除以4，得到一个固定长度句子的向量
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)), # 原来为24
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.build(input_shape=(None, 100))
# verbose 打印训练过程中的进度
model.fit(X_train, training_labels, epochs=100, validation_data=(X_test, testing_labels), verbose=1)



se = ["granny starting to fear spiders in the garden might be real",
             "game of thrones season finale showing this sunday night",
             "TensorFlow book will be a best seller"]

sequences = tokenizer.texts_to_sequences(se)
padded = pad_sequences(sequences, padding='post', truncating='post', maxlen=100)
print(model.predict(padded))