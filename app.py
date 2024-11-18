from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)  # Allow all origins for now

# Hugging Face API endpoint
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
HF_HEADERS = {
    "Authorization": "Bearer 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"
}

# Define candidate labels for classification
CANDIDATE_LABELS = ["Politics", "Sports", "Science", "Health", "Technology"]

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    try:
        # Get JSON data from the request
        data = request.json
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' in request body"}), 400
        
        text = data['text']
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "'text' must be a non-empty string"}), 400

        # Prepare payload for Hugging Face API
        payload = {
            "inputs": text,
            "parameters": {"candidate_labels": CANDIDATE_LABELS}
        }

        # Make the API request
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)
        if response.status_code != 200:
            return jsonify({"error": f"Hugging Face API error: {response.text}"}), 500

        result = response.json()

        # Prepare the response for the client
        response_data = {
            "sequence": result['sequence'],
            "top_labels": result['labels'][:2],
            "top_scores": result['scores'][:2],
        }
        return jsonify(response_data)

    except Exception as e:
        # Log and return internal server error
        print(f"Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
