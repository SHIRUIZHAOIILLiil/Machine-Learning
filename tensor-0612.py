import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.layers import Dense

#sequential只有一行，训练的神经网络只有一层
# units=1表示这一层只有一个神经元
# input_shape=[1]表示输入的形状是一个一维数组
l0 = Dense(units=1, input_shape=[1])

model = Sequential([l0])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0, 1, 2, 3, 4],dtype=np.float32)
ys = np.array([-3,-1, 1, 3, 5, 7],dtype=np.float32)
model.fit(xs, ys, epochs=500)

print(model.predict(np.array([10.0])))
print(l0.get_weights())