# 预测时间序列的技术
# 9.2.1 简单的预测来创建基线

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

ys = series[0 : splitTime], xs = range(splitTime)
forcast = series[splitTime : -1], xLabel = range(splitTime, len(series) - 1)
