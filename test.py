from flask import Flask, request, jsonify
import chromadb
import os
import requests
from groqchat import groq_chat
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# -----------------------------
# ENV VARIABLES
# -----------------------------
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

print("VOYAGE KEY LOADED:", bool(VOYAGE_API_KEY))
print("GROQ KEY LOADED:", bool(GROQ_API_KEY))


# -----------------------------
# EMBEDDING FUNCTION (SAFE)
# -----------------------------
def get_embedding(text):
    try:
        if not text:
            raise ValueError("Empty text received for embedding")

        if not VOYAGE_API_KEY:
            raise ValueError("VOYAGE_API_KEY is missing")

        response = requests.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {VOYAGE_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "input": text,
                "model": "voyage-3"
            },
            timeout=30
        )

        data = response.json()

        print("VOYAGE RESPONSE:", data)

        if "error" in data:
            raise Exception(data["error"])

        return data["data"][0]["embedding"]

    except Exception as e:
        print("🔥 EMBEDDING ERROR:", str(e))
        raise


# -----------------------------
# VECTOR DB (Chroma)
# -----------------------------
try:
    client = chromadb.PersistentClient(path="./data/vector_store")
    collection = client.get_or_create_collection(name="pdf_documents")
    print("✅ ChromaDB loaded successfully")
except Exception as e:
    print("🔥 CHROMA INIT ERROR:", str(e))
    collection = None


# -----------------------------
# RAG FUNCTION (SAFE)
# -----------------------------
def ask_rag(query, session_id):
    try:
        query_embedding = get_embedding(query)

        if collection is None:
            return "ChromaDB not initialized"

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            where={"session_id": session_id}
        )

        docs = results.get("documents", [[]])[0]

        if not docs:
            context = "No relevant context found"
        else:
            context = "\n".join(docs)

        print("Context:", context)

        prompt = f"""
Context:
{context}

Question:
{query}

Answer:
"""

        return groq_chat(prompt)

    except Exception as e:
        print("🔥 RAG ERROR:", str(e))
        return f"RAG ERROR: {str(e)}"


# -----------------------------
# ROUTE (FULL DEBUG SAFE)
# -----------------------------
@app.route("/ask", methods=["POST"])
def ask():
    try:
        print("🔥 ASK ROUTE HIT")

        data = request.get_json(force=True)
        print("REQUEST DATA:", data)

        if not data:
            return jsonify({"error": "No JSON received"}), 400

        query = data.get("query")
        session_id = data.get("session_id", "default")

        if not query:
            return jsonify({"error": "query missing"}), 400

        answer = ask_rag(query, session_id)

        return jsonify({"answer": answer})

    except Exception as e:
        print("🔥 ERROR IN /ASK:", str(e))
        return jsonify({"error": str(e)}), 500

# -----------------------------
# HEALTH CHECK ROUTE
# -----------------------------
@app.route("/")
def home():
    return "RAG server running"


# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)