import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from PIL.EpsImagePlugin import split


# # 创建数据集，从0-n-1
# dataset = tf.data.Dataset.range(10)
# # 传入参数5，指定数据集分割为5个窗口，从【0，1，2，3，4】到 【5，6，7，8，9】
# # shift指定为1每个窗口会从前一个位置移动到下一个位置
# # drop_remainder=True 靠近最后的值时，窗口小于期待值5，他们会被丢掉
# dataset = dataset.window(5, shift=1, drop_remainder=True)
# dataset = dataset.flat_map(lambda window: window.batch(5))
# dataset = dataset.map(lambda window: (window[:-1], window[-1]))
# # 打乱数据，避免顺序影响
# # 合并多个样本，提升训练效率
# dataset = dataset.shuffle(buffer_size=10)
# # prefetch 异步加载下一个批次，减少gpu等待时间
# dataset = dataset.batch(2).prefetch(1)


# 将上面的练习整合成一个函数
def windowed_dataset(series, windows_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(windows_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(windows_size + 1))
    # 打乱数据顺序，把每个window拆成一个序列和最后的1个值
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

def trend(time, slope=0.0):
    return slope * time


def seasonal_pattern(season_time):
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

# 一整个时间周期， 周期长度，振幅， 相位偏移
def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.05)
amplitude = 15
slope = 0.09
noise_level = 6

series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

series +=noise(time, noise_level=noise_level, seed=42)


splitTime = 1000
timeTrain = time[:splitTime]
x_train = series[:splitTime]
time_valid = time[splitTime:]
x_valid = series[splitTime:]
window_size = 20
buffer_size = 32
shuffle_buffer_size = 1000

dataset = windowed_dataset(series, window_size, buffer_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"),
                                    tf.keras.layers.Dense(10, activation="relu"),
                                    tf.keras.layers.Dense(1),
                                    ])

model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9))
model.fit(dataset, epochs=100, verbose=1)

# print(series[1000:1020])
print(series[1020])
# 增加一个个维度，从向量变成一维矩阵
print(model.predict(series[1000:1020][np.newaxis]))
# model.predict(series[1020])

# for feature, label in dataset.take(1):
#     print(feature)
#     print(label)