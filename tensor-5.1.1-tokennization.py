import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer


sentences = [
    'today is a sunny day',
    'today is a rainy day',
    'is it a sunny day?'
]

tokenizer = Tokenizer(num_words=100, oov_token='UNK')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

sentence = 'what is it today?'
test = tokenizer.texts_to_sequences([sentence])
print(word_index)
print(sequences)
print(test)