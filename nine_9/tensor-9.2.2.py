# 度量预测的准确率：均方误差（MSE）:简单获取在时间t的预测值和实际值之间的差，取平方然后找到所有值的平均值
# 平均绝对误差（MAE）预测值和实际值之间的绝对值相加
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

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

# plot_series(time, series)
# plt.show()

splitTime = 1000
x_valid = series[splitTime:]
naive_forecast = series[splitTime - 1 : -1]


mse_metric = tf.keras.metrics.MeanSquaredError()
mae_metric = tf.keras.metrics.MeanAbsoluteError()

mse = mse_metric(x_valid, naive_forecast).numpy()
mae = mae_metric(x_valid, naive_forecast).numpy()

# print("MSE:", mse)
# print("MAE:", mae)

# 9.2.3 移动平均值预测
# 使用移动平均值进行预测
def moving_arrange_forecast(series, window_size):
    """Forecasts the mean of the last few values.
    If window_size=1, then this is equivalent to naive forecast"""
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time : time + window_size].mean())
    return np.array(forecast)

moving_avg = moving_arrange_forecast(series, 30)[splitTime - 30:]
time_valid = time[splitTime:]
print(len(moving_avg))

mse_2 = mse_metric(x_valid, moving_avg).numpy()
mae_2 = mae_metric(x_valid, moving_avg).numpy()

print(mse_2)
print(mae_2)

plt.plot(series)
plot_series(time_valid, x_valid)
plot_series(time_valid, moving_avg)
plt.show()

