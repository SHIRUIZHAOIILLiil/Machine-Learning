# 使用Adam优化器调整学习率，降低过拟合的可能性
from read_json import read_json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences, labels, urls = read_json('./JSON-sarcasm/Sarcasm_Headlines_Dataset_v2.json')

training_size = 23000
vocab_size = 2000 # 10000
embedding_size = 7# 16

# 学习率0.001 常见值，太大容易震荡/发散，太小收敛慢，通常配合学习率衰减使用
# beta_1 0.9 一阶矩估计的指数衰减率，控制动量的影响 意味着当前梯度的90%来自过去的梯度，10%来自当前梯度
# beta_2 0.999 二阶矩估计的指数衰减率，控制自适应学习率的影响，意味着保留大量历史信息
# amsgrad False 是否使用AMSGrad变体，AMSGrad通过使用过去梯度的最大值来确保学习率不会增加，从而提高训练的稳定性
adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

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
    tf.keras.layers.Embedding(vocab_size, embedding_size),
    # 降维，假设一句话是四个词，每个词是3维向量，那么把每个词的同一向量加起来除以4，得到一个固定长度句子的向量
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(8, activation='relu'), # 原来为24
    # 添加dropout层，随机将部分神经元的输出设为0，防止过拟合
    # 但是神经元很少时，不建议使用dropout
    # tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
# model.build(input_shape=(None, 100))
# verbose 打印训练过程中的进度
model.fit(X_train, training_labels, epochs=100, validation_data=(X_test, testing_labels), verbose=1)
