from flask import Flask, request, jsonify
import chromadb
import os
import requests
from testnev import groq_chat
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# -----------------------------
# ROUTE (FULL DEBUG SAFE)
# -----------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data=request.get_json()
    userquery=data.get("query")
    answer=groq_chat(userquery)
    return jsonify({"answer":answer})

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