import json
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, jsonify
import requests
import pickle
from tensorflow.keras.models import load_model

# Ensure nltk data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)

NEWS_API_ENDPOINT = 'https://newsapi.org/v2/everything'
API_KEY = 'b9fdbdeb7c474675ab3cba163302d64e'  # Make sure to replace with your actual API key
TOPICS = ['finance', 'economy', 'stock', 'cryptocurrency']

# Load the LSTM model
model = load_model(r'C:\Users\sharm\Desktop\EDAI\financial_Pred_Model.h5')  # Update with your correct model name

# Load the tokenizer
with open(r'C:\Users\sharm\Desktop\EDAI\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)  # Update with your correct tokenizer name

# Load and preprocess news data
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

# Predict sentiment using the LSTM model
def predict_sentiment(text):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    
    # Tokenize and pad the sequences
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=50)  # Ensure this matches MAX_SEQUENCE_LENGTH used in training
    
    # Predict sentiment using the LSTM model
    prediction = model.predict(padded_sequence)
    sentiment_class = np.argmax(prediction, axis=1)  # Get the class with the highest probability
    return sentiment_class[0]  # Ensure this matches your model's output format

@app.route('/financial-news', methods=['GET'])
def get_financial_news():
    query = ' OR '.join(TOPICS)
    url = f'{NEWS_API_ENDPOINT}?q={query}&language=en&sortBy=publishedAt&apiKey={API_KEY}'
    response = requests.get(url)

    if response.status_code != 200:
        return jsonify({'error': f'Failed to retrieve news data, status code: {response.status_code}'}), response.status_code

    try:
        news_data = response.json()
    except json.JSONDecodeError:
        return jsonify({'error': 'Failed to parse JSON response'}), 500

    if news_data.get('status') != 'ok' or 'articles' not in news_data:
        return jsonify({'error': 'No articles found in the response'}), 500

    processed_news = []
    for article in news_data['articles']:
        title = article.get('title', '')
        content = article.get('content', '')

        # Ensure both title and content are strings
        if not isinstance(title, str):
            title = ''
        if not isinstance(content, str):
            content = ''

        processed_text = preprocess_text(title + ' ' + content)
        sentiment = predict_sentiment(processed_text)
        article['processed_text'] = processed_text
        article['sentiment'] = float(sentiment)  # You might need to convert it properly based on your sentiment mapping
        processed_news.append(article)

    return jsonify(processed_news)

if __name__ == "__main__":
    app.run(debug=True)
