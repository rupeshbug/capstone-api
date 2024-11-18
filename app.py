from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)

# Dynamically allow CORS based on the request's origin
allowed_origins = ["http://localhost:3000", "https://intextify.vercel.app"]
CORS(app, origins=allowed_origins)

# Load pre-trained model
classifier = pipeline("zero-shot-classification", model="distilbert-base-uncased")

# Define candidate labels
labels = ["Politics", "Sports", "Science", "Health", "Technology"]

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    try:
        # Get JSON data from the request
        data = request.json
        text = data.get('text')

        # Validate input
        if not text or not isinstance(text, str):
            return jsonify({"error": "Invalid input. Please provide a valid text."}), 400

        # Perform classification
        result = classifier(text, candidate_labels=labels)

        # Extract the top 2 labels and their probabilities
        top_labels = result['labels'][:2]
        top_scores = result['scores'][:2]

        # Format response
        response = {
            "sequence": result['sequence'],
            "top_labels": top_labels,
            "top_scores": top_scores,
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
