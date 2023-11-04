from flask import Flask, render_template
import requests

import numpy as np


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
app = Flask(__name__)



# Replace this with your actual News API key
NEWS_API_KEY = "9a2615095e6c414cb2a87d1858b8aa8c"
NEWS_API_URL = "https://newsapi.org/v2/top-headlines?sources=bbc-news"
class_labels = ['Business', 'Entertainment', 'Politics', 'Sport',' Tech']
# the maximum number of words to keep, based on word frequency
NUM_WORDS = 1000

# dimension of the dense embedding that will be used in the embedding layer of the model
EMBEDDING_DIM = 16

# maximum length of all sequences
MAXLEN = 120

# padding strategy
PADDING = 'post'

# token to replace out-of-vocabulary words during text_to_sequence() calls
OOV_TOKEN = "<OOV>"

# proportion of data used for training
TRAINING_SPLIT = .8

@app.route('/')
def index():
    response = requests.get(NEWS_API_URL, headers={"X-Api-Key": NEWS_API_KEY})
    news_data = response.json()

    articles = news_data['articles']
    test_sentences = [article['description'] for article in articles if article.get('description')]  # Adjust based on your data

    # Tokenization and padding
    test_tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token=OOV_TOKEN)
    test_tokenizer.fit_on_texts(test_sentences)
    test_word_index = test_tokenizer.word_index

    test_sequences = test_tokenizer.texts_to_sequences(test_sentences)
    test_padded_seq = pad_sequences(test_sequences, maxlen=MAXLEN, padding=PADDING)

    # Load your trained model
    model = load_model('textclassify.h5')  # Replace with the path to your trained model

    # Make predictions on the test data
    predictions = model.predict(test_padded_seq)
    predicted_classes = predictions.argmax(axis=1)
    for article, prediction in zip(articles, predicted_classes):
        article['predicted_class'] = class_labels[prediction]
    for article in articles:
        # Add the 'image_url' key to each article with the image URL
        article['image_url'] = article.get('urlToImage', 'URL for image not available')
    

    return render_template('index.html', articles=articles)

if __name__ == '__main__':
    app.run(debug=True)
