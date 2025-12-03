from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import sys
import logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Load model and vectorizer
MODEL_PATH = "models/nb_model.pkl"
TFIDF_PATH = "models/tfidf.pkl"

try:
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
except Exception as e:
    logging.exception("Failed to load model or vectorizer.")
    sys.exit(1)

# Use string keys because labels may be saved as strings
category_map = {"1": "World", "2": "Sports", "3": "Business", "4": "Sci/Tech"}

@app.route("/", methods=["GET"])
def home():
    return "✅ Flask API is running! Use POST /predict with JSON {\"text\": \"...\"}."

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON."}), 400

    data = request.get_json()
    text = (data or {}).get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Empty or invalid 'text' field."}), 400

    try:
        text_tfidf = tfidf.transform([text])
        pred = model.predict(text_tfidf)[0]
        pred_str = str(pred)
        # try to convert to int for numeric clients, otherwise return string
        try:
            pred_int = int(pred_str)
        except Exception:
            pred_int = None

        category = category_map.get(pred_str, "Unknown")
        resp = {"predicted_class": pred_int if pred_int is not None else pred_str, "category": category}
        return jsonify(resp)
    except Exception as e:
        logging.exception("Prediction error")
        return jsonify({"error": "Prediction failed.", "details": str(e)}), 500

if __name__ == "__main__":
    # Run from backend/ directory: python app.py
    app.run(host="0.0.0.0", port=5000, debug=True)