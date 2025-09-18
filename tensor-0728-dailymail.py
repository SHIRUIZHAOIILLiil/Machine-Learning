import tensorflow as tf
import tensorflow_datasets as tfds

# catsAndDogs = tfds.load("cats_vs_dogs", split="train", as_supervised=True)

cnnDailyMail = tfds.load("cnn_dailymail", split="train", as_supervised=True)