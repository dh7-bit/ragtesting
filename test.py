from flask import Flask,request,jsonify
from sentence_transformers import SentenceTransformer
import chromadb
import os
import requests

app = Flask(__name__)

# -----------------------------
# 1. LOAD MODELS
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# GROQ USING REQUESTS (YOUR CODE)
# -----------------------------
GROQ_API_KEY = os.getenv('Groq_key')

def groq_chat(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


# -----------------------------
# 2. VECTOR DB (Chroma)
# -----------------------------
client = chromadb.PersistentClient(path="./data/vector_store")
collection = client.get_or_create_collection(name="pdf_documents")

# -----------------------------
# 3. QUERY FUNCTION
# -----------------------------
def ask_rag(query, session_id="5bc83d0c-0add-4054-bcb5-6eaa7aaa46db"):

    # -------------------------
    # Step 1: Embed query
    # -------------------------
    query_embedding = embedding_model.encode(query).tolist()

    # -------------------------
    # Step 2: Search vector DB
    # -------------------------
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where={"session_id": session_id}
    )

    docs = results["documents"][0]

    context = "\n".join(docs)
    print("printing context"+context)

    # -------------------------
    # Step 3: Build prompt
    # -------------------------
    prompt = f"""
there is context that i am providing you and please answer and summarise 
the answer by seeing user query what user want and then check in context  and then
provide answer by using context

Context:
{context}

Question:
{query}

Answer
"""

    # -------------------------
    # Step 4: Call Groq LLM (UPDATED)
    # -------------------------
    response = groq_chat(prompt)

    return response


# -----------------------------
# 4. FLASK ROUTES
# -----------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query")
    session_id = data.get("session_id", "5bc83d0c-0add-4054-bcb5-6eaa7aaa46db")

    answer = ask_rag(query, session_id)
    return jsonify({"answer": answer})


# -----------------------------
# 5. RUN SERVER
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render provides PORT
    app.run(host="0.0.0.0", port=port)