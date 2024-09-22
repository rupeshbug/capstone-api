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

# Dictionary to describe topics and their keywords
topic_descriptions = {
    0: {"name": "Music and Media", "keywords": ["music", "media", "album", "concert", "artist", "song", "band", "vocalist", "video", "radio", "performance", "entertainment"]},
    1: {"name": "Science and Technology", "keywords": ["science", "technology", "innovation", "research", "experiment", "theory", "biology", "chemistry", "physics", "data", "AI", "machine learning", "engineering"]},
    2: {"name": "Politics and Society", "keywords": ["politics", "government", "election", "policy", "society", "people", "community", "rights", "debate", "campaign", "democracy", "activism"]},
    3: {"name": "Miscellaneous News and Events", "keywords": ["news", "event", "report", "update", "story", "headline", "coverage", "journalism", "reporter", "article", "current events", "features"]},
    4: {"name": "Health and Diseases", "keywords": ["health", "disease", "virus", "treatment", "symptoms", "healthcare", "medical", "pandemic", "nutrition", "exercise", "wellness", "prevention"]},
}

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get('text')
    load_models()
    cleaned_text = preprocess(text)
    transformed_text = tfidf_vectorizer.transform([cleaned_text])
    topics = lda_model.transform(transformed_text)

    # Get the two highest topics with their keywords
    result = []
    for i in topics[0].argsort()[-2:][::-1]:
        result.append({
            "topic": topic_descriptions[i]["name"],
            "score": topics[0][i],
            "keywords": topic_descriptions[i]["keywords"]
        })
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
