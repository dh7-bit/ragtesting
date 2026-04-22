import os 
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY=os.getenv('Groq_key')
import requests
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

    response = requests.post(url, headers=headers, json=payload,timeout=30)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]