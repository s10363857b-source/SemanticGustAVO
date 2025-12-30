# GustAVO Chatbot dell'IIS Avogadro di Torino
# versione con EMBEDDING semantici (no TF-IDF, no rete neurale)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import random
import os

from sentence_transformers import SentenceTransformer, util
import torch

# =========================
# Flask setup
# =========================
app = Flask(__name__, static_folder=".")
CORS(app)

# =========================
# Carica intents.json
# =========================
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)["intents"]

# =========================
# Carica modello embedding
# =========================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# Prepara risposte e embedding
# =========================
responses = {}
intent_embeddings = []

for intent in intents:
    tag = intent["tag"]
    responses[tag] = intent["responses"]

    for pattern in intent["patterns"]:
        intent_embeddings.append({
            "tag": tag,
            "text": pattern,
            "embedding": embedding_model.encode(
                pattern, convert_to_tensor=True
            )
        })

# =========================
# Cronologia conversazioni
# =========================
conversations = {}     # session_id -> lista messaggi
N_HISTORY = 2          # messaggi precedenti usati come contesto
MAX_HISTORY = 10       # massimo messaggi salvati

# =========================
# Classificazione intent con embedding
# =========================
def classify_intent_embedding(user_message, threshold=0.5):
    user_embedding = embedding_model.encode(
        user_message, convert_to_tensor=True
    )

    best_score = 0
    best_tag = None

    for item in intent_embeddings:
        score = util.cos_sim(
            user_embedding, item["embedding"]
        ).item()

        if score > best_score:
            best_score = score
            best_tag = item["tag"]

    if best_score < threshold:
        return None, best_score

    return best_tag, best_score


def generate_response(intent_tag):
    if intent_tag in responses:
        return random.choice(responses[intent_tag])
    return "Non ho capito bene, puoi riformulare?"

# =========================
# Static files (immagini)
# =========================
@app.route('/img/<path:filename>')
def serve_images(filename):
    try:
        return send_from_directory('img', filename)
    except FileNotFoundError:
        return jsonify({"error": "Image not found"}), 404


# =========================
# Debug files
# =========================
@app.route("/debug-files")
def debug_files():
    img_files = []
    if os.path.exists('img'):
        img_files = os.listdir('img')
    return jsonify({
        "img_folder_exists": os.path.exists('img'),
        "images_in_img_folder": img_files,
        "current_directory": os.getcwd()
    })


# =========================
# Home page
# =========================
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


# =========================
# Test route
# =========================
@app.route("/test", methods=["GET"])
def test():
    return jsonify({
        "status": "ok",
        "message": "GustAVO (embedding version) è attivo!"
    })


# =========================
# Chat endpoint
# =========================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    if not user_message:
        return jsonify({"answer": "Per favore scrivi qualcosa."})

    # Recupera cronologia
    history = conversations.get(session_id, [])
    history.append({"role": "user", "text": user_message})

    # Costruisci contesto
    prev_texts = [
        m["text"]
        for m in history[-N_HISTORY:]
        if m["role"] == "user"
    ]
    contextual_message = " ".join(prev_texts)

    # Classifica intent con embedding
    intent, confidence = classify_intent_embedding(
        contextual_message
    )

    if intent is None:
        bot_reply = "Non ho capito bene, puoi riformulare?"
    else:
        bot_reply = generate_response(intent)

    history.append({"role": "bot", "text": bot_reply})
    conversations[session_id] = history[-MAX_HISTORY:]

    return jsonify({
        "answer": bot_reply,
        "intent": intent,
        "confidence": round(float(confidence), 2),
        "history": conversations[session_id]
    })


# =========================
# Avvio app
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
