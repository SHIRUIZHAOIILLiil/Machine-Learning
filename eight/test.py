# -*- coding: utf-8 -*-
# 基于字符的编码（WSL + GPU优化版）

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# —— 可选：减少日志噪音（2=只显示ERROR；不需要可删）
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# —— 1) GPU 显存“按需分配”，避免一次性吃满
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

# —— 2) 开启混合精度（新显卡上能显著提速）
mixed_precision.set_global_policy("mixed_float16")

def readTxtFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as f:
        return f.read()

def predict_class(model, x):
    # x: (batch=1, block_size) 的整数序列
    probs = model.predict(x, verbose=0)  # (1, totalChars)
    return int(np.argmax(probs, axis=-1)[0])

# —— 读取数据并做字符级编码
data = readTxtFile("./XW_ab.txt")

# 注意：你是“字符级”，所以不要 split 成词
tokenizer = Tokenizer(char_level=True, filters='', lower=False)
tokenizer.fit_on_texts([data])
ids = tokenizer.texts_to_sequences([data])[0]     # 整数序列
totalChars = len(tokenizer.word_index) + 1        # 含 0（padding）

# —— 构造训练样本（长度 block_size → 预测下一个字符）
block_size = 100
xs, ys = [], []
for i in range(0, len(ids) - block_size):
    x = ids[i:i + block_size]         # 长度=block_size
    y = ids[i + block_size]           # 下一个字符
    xs.append(x)
    ys.append(y)

xs = np.array(xs, dtype=np.int32)
ys = tf.keras.utils.to_categorical(np.array(ys, dtype=np.int32), num_classes=totalChars)

# —— 用 tf.data 提升吞吐
BATCH = 512
ds = tf.data.Dataset.from_tensor_slices((xs, ys)).shuffle(10000).batch(BATCH).prefetch(tf.data.AUTOTUNE)

# —— 模型（注意：LSTM 单元数别设成 totalChars-1，会又慢又占显存）
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=totalChars, output_dim=128, name="emb"),      # 适当加大维度更有效
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, dropout=0.1), name="bi_lstm"),# 256/512 视显存调
    tf.keras.layers.Dense(totalChars, activation='softmax', dtype='float32')          # 混合精度下最后一层回到 float32
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("TF:", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))
history = model.fit(ds, epochs=25, verbose=1, batch_size=BATCH)

# —— 文本生成（字符级）
# 小贴士：字符级拼接不需要在字符之间额外加空格（除非预测到的就是空格）
seed_text = "You know nothing, John Snow."
next_steps = 100

# 反向索引：id → 字符
index_word = {idx: ch for ch, idx in tokenizer.word_index.items()}
index_word[0] = ''   # padding

# 保证初始输入长度为 block_size（不够就左侧 pad 0）
seed_ids = tokenizer.texts_to_sequences([seed_text])[0]
seed_ids = seed_ids[-block_size:]  # 截断到最多 block_size
seed_ids = [0] * (block_size - len(seed_ids)) + seed_ids

for _ in range(next_steps):
    x = np.array([seed_ids], dtype=np.int32)     # (1, block_size)
    pred_id = predict_class(model, x)
    next_char = index_word.get(pred_id, '')
    seed_text += next_char                       # 直接拼接字符
    # 维护长度为 block_size 的滑动窗口
    seed_ids = seed_ids[1:] + [pred_id]

print(seed_text)
