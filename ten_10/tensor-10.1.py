import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# 创建数据集，从0-n-1
dataset = tf.data.Dataset.range(10)
# 传入参数5，指定数据集分割为5个窗口，从【0，1，2，3，4】到 【5，6，7，8，9】
# shift指定为1每个窗口会从前一个位置移动到下一个位置
# drop_remainder=True 靠近最后的值时，窗口小于期待值5，他们会被丢掉
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1]))

dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)
