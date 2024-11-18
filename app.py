from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load pre-trained BERT model fine-tuned for classification
classifier = pipeline("zero-shot-classification", model="distilbert-base-uncased")

# Define candidate labels for classification (e.g., Politics, Sports, Science)
labels = ["Politics", "Sports", "Science", "Health", "Technology"]

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    try:
        # Get JSON data from the request
        data = request.json
        text = data.get('text')
        
        # Ensure the text field is provided and is not empty
        if not text or not isinstance(text, str):
            return jsonify({"error": "Invalid input. Please provide a valid text."}), 400
        
        # Get classification results
        result = classifier(text, candidate_labels=labels)
        
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
    
    except Exception as e:
        # Return an error message if something goes wrong
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
