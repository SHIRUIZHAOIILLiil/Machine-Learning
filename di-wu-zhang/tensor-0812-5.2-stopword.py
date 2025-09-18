from bs4 import BeautifulSoup

sentences = [
    'today is a sunny day',
    'today is a rainy day',
    'is it a sunny day?'
    'I really enjoyed walking in the snow today',
]

# Example sentence with HTML tags
# 去除html标签
sentence = "<br>what is it today? it is a windy</br>"
soup = BeautifulSoup(sentence, features="html.parser")
sentence = soup.get_text()

print(sentence)


stopwords = ['a', 'about', 'above']

# 去除停用词
words = sentence.split()
filtered_sentence = ""
for word in words:
    if word not in stopwords:
        filtered_sentence = filtered_sentence + word + " "

sentences.append(filtered_sentence)

print(sentences)