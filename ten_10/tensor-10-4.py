import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# 10.4 使用神经网络进行时间序列预测整体

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

forecast = []
for time in range(len(series) - window_size):
    forecast.append(model.predict(series[time : time + window_size][np.newaxis]))

forecast = forecast[splitTime - window_size:]
# [:,0, 0] 压缩维度，把多余的维度去掉，简单写法
result = np.array(forecast)[:, 0, 0]

mae_metric = tf.keras.metrics.MeanAbsoluteError()
mae = mae_metric(x_valid, result).numpy()

print(mae)
