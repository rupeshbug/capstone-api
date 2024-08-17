from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load models once
lda_model = None
tfidf_vectorizer = None

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def load_models():
    global lda_model, tfidf_vectorizer
    if lda_model is None or tfidf_vectorizer is None:
        with open('lda_model.pkl', 'rb') as f:
            lda_model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)

# Dictionary to describe topics
topic_descriptions = {
    0: "Music and Media",
    1: "Science",
    2: "Politics and Society",
    3: "Miscellaneous News and Events",
    4: "Health and Diseases",
}

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get('text')
    load_models()
    cleaned_text = preprocess(text)
    transformed_text = tfidf_vectorizer.transform([cleaned_text])
    topics = lda_model.transform(transformed_text)
    result = [(topic_descriptions[i], topics[0][i]) for i in topics[0].argsort()[-2:][::-1]]
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
