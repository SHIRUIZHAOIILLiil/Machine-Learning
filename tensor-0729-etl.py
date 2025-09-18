import tensorflow as tf
import tensorflow_datasets as tfds
import multiprocessing

# train_data = tfds.load('cats_vs_dogs', split='train', as_supervised=True)
#
# file_pattern = '/Users/anmoli/tensorflow_datasets/cats_vs_dogs/4.0.1/cats_vs_dogs-train.tfrecord*'
# files = tf.data.Dataset.list_files(file_pattern)
#
# train_dataset = files.interleave(
#                     tf.data.TFRecordDataset,
#                     cycle_length=4,
#                     num_parallel_calls=tf.data.experimental.AUTOTUNE
# )
#
# def read_tfrecord(serialized_example):
#     feature_description = {
#         'image': tf.io.FixedLenFeature([], tf.string, ""),
#         'label': tf.io.FixedLenFeature([], tf.int64, -1),
#     }
#     example = tf.io.parse_single_example(serialized_example, feature_description)
#     image = tf.io.decode_jpeg(example['image'], channels=3)
#     image = tf.cast(image, tf.float32)
#     image = image / 255.0
#     image = tf.image.resize(image, (300, 300))
#     return image, example['label']
#
# cores = multiprocessing.cpu_count()
# print(cores)
# train_dataset = train_dataset.map(read_tfrecord, num_parallel_calls=cores)
# train_dataset = train_dataset.cache()
#
# train_dataset = train_dataset.shuffle(1024).batch(32)
# train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
#
# model = tf.keras.models.Sequential([
#     tf.keras.layers.InputLayer(input_shape=(300, 300, 3)),
#     # 提取32个特征图，卷积核大小为3x3，激活函数为ReLU，一张图用32个卷积核进行评估提取32个特征
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#     # 池化层，池化大小为2x2， 每张图修剪为原来的一半，
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     # 从新提取的图上重新提取
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     # 二维特征图转为一维特征图
#     tf.keras.layers.Flatten(),
#     # 包含所有特征的全连接层
#     tf.keras.layers.Dense(128, activation=tf.nn.relu),
#     # 防止过拟合，随即屏蔽50%的神经元
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
#
# model.compile(optimizer='sgd',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(train_dataset, epochs=10, verbose=1)


train_data = tfds.load('cats_vs_dogs', split='train[:80%]', as_supervised=True)
val_data = tfds.load('cats_vs_dogs', split='train[80%:]', as_supervised=True)

def preprocess(image, label):
    image = tf.image.resize(image, (150, 150))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# 应用处理 & 构建输入管道
train_ds = train_data.map(preprocess).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_data.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(150, 150, 3)),
    # 提取32个特征图，卷积核大小为3x3，激活函数为ReLU，一张图用32个卷积核进行评估提取32个特征
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    # 池化层，池化大小为2x2， 每张图修剪为原来的一半，
    tf.keras.layers.MaxPooling2D((2, 2)),
    # 从新提取的图上重新提取
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    # 二维特征图转为一维特征图
    tf.keras.layers.Flatten(),
    # 包含所有特征的全连接层
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    # 防止过拟合，随即屏蔽50%的神经元
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, epochs=10, verbose=1)


