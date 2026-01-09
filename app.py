# GustAVO Chatbot dell'IIS Avogadro di Torino
# versione con EMBEDDING semantici (no TF-IDF, no rete neurale)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import random
import os
import faiss
import sys

from sentence_transformers import SentenceTransformer

# =========================
# Flask setup
# =========================
app = Flask(__name__, static_folder=".")
CORS(app)

# =========================
# Carica intents.json
# =========================
with open(os.path.dirname(os.path.abspath(__file__)) + "/intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)["intents"]

responses = {}
for intent in intents:
    responses[intent["tag"]] = intent["responses"]
# =========================
# Carica modello embedding
# =========================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# Prepara risposte e embedding
# =========================
FAISS_INDEX_PATH = os.path.dirname(os.path.abspath(__file__)) + "/intents.faiss"
METADATA_PATH = os.path.dirname(os.path.abspath(__file__)) + "/intents_meta.json"
# Se il file contenente emebedding esiste già, lo si carica evitando sprechi di risorse, sennò lo si crea sul momento generando un file index.faiss e un intents_meta.json
if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
    print("Caricamento indice FAISS esistente")

    faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
else:
    print("Indice FAISS non trovato, creazione indice FAISS...")

    texts = []
    metadata = []

    responses = {}

    for intent in intents:
        tag = intent["tag"]
        responses[tag] = intent["responses"]

        for pattern in intent["patterns"]:
            texts.append(pattern)
            metadata.append({
                "tag": tag,
                "text": pattern
            })

    # Embedding
    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype("float32")

    # Dimensione embedding
    dim = embeddings.shape[1]

    # Indice FAISS (cosine similarity)
    faiss_index = faiss.IndexFlatIP(dim)

    # Normalizzazione per cosine similarity
    faiss.normalize_L2(embeddings)

    # Aggiunta vettori
    faiss_index.add(embeddings)

    # Salvataggio su disco
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Indice FAISS creato e salvato!")

# =========================
# Cronologia conversazioni
# =========================
conversations = {}     # session_id -> lista messaggi
N_HISTORY = 2          # messaggi precedenti usati come contesto
MAX_HISTORY = 10       # massimo messaggi salvati

# =========================
# Classificazione intent con embedding
# =========================
def classify_intent_embedding(user_message, threshold=0.7):
    user_embedding = embedding_model.encode(
        [user_message]
    ).astype("float32")

    faiss.normalize_L2(user_embedding)

    scores, indices = faiss_index.search(user_embedding, k=1)

    best_score = scores[0][0]
    best_idx = indices[0][0]

    if best_score < threshold:
        return None, best_score

    best_tag = metadata[best_idx]["tag"]
    return best_tag, best_score

def generate_response(intent_tag):
    if intent_tag in responses:
        return random.choice(responses[intent_tag])
    return "Non ho capito bene, puoi riformulare?"

# =========================
# Logica chatbot
# =========================

def chatbot_logic(user_message, session_id="default"):
    if not user_message:
        return "Per favore scrivi qualcosa.", None, 0.0

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

    # Classifica intent
    intent, confidence = classify_intent_embedding(contextual_message)

    if intent is None:
        bot_reply = "Non ho capito bene, puoi riformulare?"
    else:
        bot_reply = generate_response(intent)

    history.append({"role": "bot", "text": bot_reply})
    conversations[session_id] = history[-MAX_HISTORY:]

    return bot_reply, intent, confidence


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
        "message": "attivo!"
    })


# =========================
# Chat endpoint
# =========================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    bot_reply, intent, confidence = chatbot_logic(
        user_message, session_id
    )

    return jsonify({
        "answer": bot_reply,
        "intent": intent,
        "confidence": round(float(confidence), 2),
        "history": conversations.get(session_id, [])
    })



# =========================
# Avvio app
# =========================

if __name__ == "__main__":


    if "--console" in sys.argv:
        print("WARNING: Chatbot in modalità console")
        print("WARNING: In questa modalità, il chatbot si interfaccia DIRETTAMENTE con la CONSOLE e quindi NON ARRIVERA AL FRONTEND")
        while True:
            user_input = input("Tu: ")
            if user_input.lower() in ("exit", "quit"):
                break

            reply, intent, confidence = chatbot_logic(user_input)
            print(f"Bot: {reply}")
            print(f"    ↳ intent={intent}, conf={confidence:.2f}\n")
    else:
        port = int(os.environ.get("PORT", 5000))

        app.run(host="0.0.0.0", port=port)



