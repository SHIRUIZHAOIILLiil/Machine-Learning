import tensorflow as tf
import tensorflow_datasets as tfds
mnist_data = tfds.load("fashion_mnist")
mnist_train = tfds.load("fashion_mnist", split="train")

# assert isinstance(mnist_train, tf.data.Dataset)
# print(type(mnist_train))

# for item in mnist_train.take(1):
#     print(type(item))
#     print(item.keys())

mnist_test, info = tfds.load("fashion_mnist", with_info="true")
print(info)