from flask import Flask, request, jsonify
import chromadb
import os
import requests
from groqchat import groq_chat
app = Flask(__name__)

# -----------------------------
# ENV VARIABLES
# -----------------------------
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
HF_API_KEY = os.getenv("HF_TOKEN")
from fastembed import TextEmbedding

# lightweight model (production-safe)
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def get_embedding(text):
    return list(embedding_model.embed([text]))[0].tolist()

# -----------------------------
# 3. VECTOR DB (Chroma)
# -----------------------------
client = chromadb.PersistentClient(path="./data/vector_store")
collection = client.get_or_create_collection(name="pdf_documents")

# -----------------------------
# 4. QUERY FUNCTION
# -----------------------------
def ask_rag(query, session_id="5bc83d0c-0add-4054-bcb5-6eaa7aaa46db"):

    # Step 1: Get embedding from API
    query_embedding = get_embedding(query)

    # Step 2: Search vector DB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where={"session_id": session_id}
    )

    docs = results["documents"][0]
    context = "\n".join(docs)

    print("Context:", context)

    # Step 3: Prompt
    prompt = f"""
there is context that i am providing you and please answer and summarise 
the answer by seeing user query what user want and then check in context  and then
provide answer by using context


Context:
{context}

Question:
{query}

Answer:
"""

    # Step 4: LLM
    return groq_chat(prompt)

# -----------------------------
# 5. ROUTE
# -----------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query")
    session_id = data.get("session_id", "default")

    answer = ask_rag(query, session_id)
    return jsonify({"answer": answer})

# -----------------------------
# 6. RUN SERVER
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)