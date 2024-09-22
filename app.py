from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords

# Download the stopwords corpus
nltk.download('stopwords')

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

# Dictionary to describe topics with keywords
topic_keywords = {
    0: {
        "description": "Music and Media",
        "keywords": ["music", "concert", "album", "singer", "band", "performance", "song", "guitar", "festival", "vocalist", "lyrics", "poem", "actors", "films", "movies"]
    },
    1: {
        "description": "Science and Technology",
        "keywords": ["science", "research", "experiment", "theory", "study", "data", "technology", "innovation", "scientist", "discovery", "space", "physics", "matter", "software", "artificial intelligence", "computer", "mobile"]
    },
    2: {
        "description": "Politics and Society",
        "keywords": ["politics", "government", "election", "policy", "president", "party", "voter", "campaign", "legislation", "society"]
    },
    3: {
        "description": "Miscellaneous News and Events",
        "keywords": ["news", "event", "update", "report", "breaking", "headline", "story", "coverage", "announcement", "feature"]
    },
    4: {
        "description": "Health and Diseases",
        "keywords": ["health", "disease", "virus", "treatment", "medicine", "epidemic", "research", "patient", "doctor", "healthcare", "food", "nurse", "hospital", "clinic"]
    },
    5: {
        "description": "Sports",
        "keywords": ["sports", "football", "basketball", "soccer", "tournament", "athlete", "team", "game", "score", "championship", "players", "goals", "runs", "win", "lose"]
    },
}

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get('text')
    load_models()
    cleaned_text = preprocess(text)
    transformed_text = tfidf_vectorizer.transform([cleaned_text])
    topics = lda_model.transform(transformed_text)

    # Adjust scores based on keywords
    adjusted_scores = topics[0].copy()
    for idx, topic in topic_keywords.items():
        keyword_weight = sum([cleaned_text.count(kw) for kw in topic['keywords']])
        adjusted_scores[idx] += keyword_weight

    # Return the top topics based on adjusted scores
    result = [(topic_keywords[i]["description"], adjusted_scores[i]) for i in adjusted_scores.argsort()[-2:][::-1]]
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
