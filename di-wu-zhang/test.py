import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
import string
import nltk
from nltk.corpus import stopwords

# 初始化
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text_bytes):
    text = str(text_bytes).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    return ' '.join(word for word in words if word not in stop_words)

# 高效加载数据
train_data = tfds.as_numpy(tfds.load(
    'imdb_reviews',
    split='train',
    batch_size=1024,
    as_supervised=True
))

# 并行预处理
imdb_sentences = [preprocess_text(text) for text, _ in train_data]

# Tokenizer处理
tokenizer = Tokenizer(num_words=25000)
tokenizer.fit_on_texts(imdb_sentences)
sequences = tokenizer.texts_to_sequences(imdb_sentences)


print(tokenizer.word_index)