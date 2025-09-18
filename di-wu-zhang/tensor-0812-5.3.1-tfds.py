import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from bs4 import BeautifulSoup
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# print(stop_words)

# str.maketrans(x, y, z)
# x：需要被替换的字符（此处为空字符串 ''，表示不替换任何字符）。
# y：替换为的字符（此处为空字符串 ''，同上）。
# z：需要删除的字符（此处为 string.punctuation，即所有标点符号）。
table = str.maketrans('', '', string.punctuation)

imdb_sentences = []
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split='train'))
for item in train_data:
    sentence = str(item['text']).lower()
    soup = BeautifulSoup(sentence, features="html.parser")
    sentence = soup.get_text()
    words = sentence.split()
    filtered_sentence = " "
    for word in words:
        word = word.translate(table)
        if word not in stop_words:
            filtered_sentence += filtered_sentence + word + " "
    imdb_sentences.append(filtered_sentence)


tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=25000)
tokenizer.fit_on_texts(imdb_sentences)
sequences = tokenizer.texts_to_sequences(imdb_sentences)

print(tokenizer.word_index)