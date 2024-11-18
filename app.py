from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load pre-trained BERT model fine-tuned for classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define candidate labels for classification (e.g., Politics, Sports, Science)
labels = ["Politics", "Sports", "Science", "Health", "Technology"]

# Function to preprocess the text (if needed)
def preprocess(text):
    # Optionally, apply text cleaning/preprocessing here
    return text

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get('text')
    
    # Preprocess the text if needed
    cleaned_text = preprocess(text)
    
    # Get classification results
    result = classifier(cleaned_text, candidate_labels=labels)
    
    # Get the top 2 labels and their probabilities
    top_labels = result['labels'][:2]
    top_scores = result['scores'][:2]
    
    # Format the result as desired
    response = {
        "sequence": result['sequence'],
        "top_labels": top_labels,
        "top_scores": top_scores
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
