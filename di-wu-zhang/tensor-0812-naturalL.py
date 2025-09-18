import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'today is a sunny day',
    'today is a rainy day',
    'is it a sunny day?'
    'I really enjoyed walking in the snow today',
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
# padding 填充，使得句子有相同长短的形状
padded = pad_sequences(sequences)

post_padding = pad_sequences(sequences, padding='post')
print(padded)
print(post_padding)

# 会从前面截断，丢失前面的单词
maxlenPadding = pad_sequences(sequences, padding='post', maxlen=5)
print()
print(maxlenPadding)

truncatingPadding = pad_sequences(sequences, padding='post', maxlen=5, truncating='post')
tp = pad_sequences(sequences, maxlen=10)
print()
print(truncatingPadding)

print()
print(tp)



