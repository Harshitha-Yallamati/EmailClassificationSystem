"""Flask application for the Email Classification System."""
from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "logistic_model.pkl"
VECTORIZER_PATH = ROOT / "tfidf_vectorizer.pkl"

app = Flask(__name__)

# Global objects loaded once at startup.
MODEL = None
VECTORIZER = None


# ============================
# LOAD MODEL & VECTORIZER
# ============================
def load_artifacts() -> None:
    global MODEL, VECTORIZER

    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        missing = [
            p.name for p in (MODEL_PATH, VECTORIZER_PATH) if not p.exists()
        ]
        raise FileNotFoundError(
            "Missing required artifacts: " + ", ".join(missing)
            + ". Train the model and save .pkl files first."
        )

    with MODEL_PATH.open("rb") as model_file:
        MODEL = pickle.load(model_file)

    with VECTORIZER_PATH.open("rb") as vec_file:
        VECTORIZER = pickle.load(vec_file)


# ============================
# TEXT PREPROCESSING
# ============================
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [
        token for token in text.split()
        if token not in ENGLISH_STOP_WORDS and len(token) > 2
    ]
    return " ".join(tokens)


# ============================
# ✅ PERFECT HYBRID CLASSIFIER
# ============================
def classify_email(message: str) -> str:
    """
    Hybrid classification:
    1. Rule-based override (High-confidence keywords)
    2. ML-based fallback (TF-IDF + Logistic Regression)
    """

    text = message.lower()

    # ✅ RULE-BASED OVERRIDES (HIGH CONFIDENCE)
    if any(word in text for word in ["urgent", "immediate", "asap", "server down", "security alert"]):
        return "Important"

    if any(word in text for word in ["offer", "discount", "sale", "deal", "buy now", "limited time"]):
        return "Promotion"

    if any(word in text for word in ["free", "win", "winner", "prize", "click here", "claim now"]):
        return "Spam"

    # ✅ FALLBACK TO ML MODEL
    if MODEL is None or VECTORIZER is None:
        raise RuntimeError("Model artifacts not loaded.")

    cleaned = preprocess_text(message)
    transformed = VECTORIZER.transform([cleaned])
    prediction = MODEL.predict(transformed)

    return prediction[0]


# ============================
# ROUTES
# ============================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)

    if not payload:
        payload = {"message": request.form.get("message", "")}

    message = (payload.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Email text is required."}), 400

    try:
        label = classify_email(message)
        return jsonify({"label": label})

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500

    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


# ============================
# APP ENTRY POINT
# ============================
if __name__ == "__main__":
    load_artifacts()
    port = int(os.environ.get("PORT", 8001))
    app.run(host="0.0.0.0", port=port, debug=True)
