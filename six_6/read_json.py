import json
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.corpus import stopwords
import string

def read_json(filename):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    table = str.maketrans('', '', string.punctuation)


    sentences = []
    labels = []
    urls = []
    with open(filename) as json_file:
        for line in json_file:
            item = json.loads(line)
            sentence = item['headline'].lower()
            sentence = sentence.replace(',', ' , ')
            sentence = sentence.replace('.', ' . ')
            sentence = sentence.replace('-', ' - ')
            sentence = sentence.replace('/', ' / ')
            soup = BeautifulSoup(sentence, features="html.parser")
            sentence = soup.get_text()
            words = sentence.split()
            filtered_sentence = ""
            tokens = []
            for word in words:
                word = word.translate(table)
                if word not in stop_words:
                    tokens.append(word)
            filtered_sentence += " ".join(tokens)
            sentences.append(filtered_sentence)
            labels.append(item['is_sarcastic'])
            urls.append(item['article_link'])
    return sentences, labels, urls


if __name__ == "__main__":
    sentences, labels, urls = read_json('./JSON-sarcasm/Sarcasm_Headlines_Dataset_v2.json')
    vocabulary_size = 20000
    maxlength = 10
    truncType = 'post'
    paddingType = 'post'
    oov_tok = "<OOV>"

    training_size = 23000

    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    tokenizer = Tokenizer(num_words=vocabulary_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    trainingSequences = tokenizer.texts_to_sequences(training_sentences)
    padded = pad_sequences(trainingSequences, padding=paddingType)

    print(word_index)