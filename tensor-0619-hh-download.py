import urllib.request
import zipfile
url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"
filename = "horse-or-human.zip"
training_dir = "horse-or-human/training"
urllib.request.urlretrieve(url, filename)

zip_ref = zipfile.ZipFile(filename, "r")
zip_ref.extractall(training_dir)
zip_ref.close()