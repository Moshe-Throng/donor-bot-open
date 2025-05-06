# app.py

from flask import Flask, request
import requests
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

load_dotenv()  # Load variables from .env

app = Flask(__name__)

# Telegram and OpenAI setup
BOT_TOKEN = os.environ.get("BOT_TOKEN")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

# Load OpenAI-powered vector index (built from EU donor PDF)
embeddings = OpenAIEmbeddings()
vector_db = FAISS.load_local("eu_vector_index", embeddings, allow_dangerous_deserialization=True)
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=vector_db.as_retriever(),
    chain_type="stuff"
)

@app.route('/')
def home():
    return "Bot is running with GPT + EU document search!"

@app.route(f"/webhook/{BOT_TOKEN}", methods=["POST"])
def webhook():
    try:
        data = request.get_json()
        if "message" not in data:
            return {"ok": True}

        chat_id = data["message"]["chat"]["id"]
        text = data["message"].get("text", "").strip()

        if text.startswith("/donor"):
            try:
                _, donor, country = text.split(" ", 2)
                question = f"What is {donor}'s current development strategy in {country}?"
            except ValueError:
                reply = "Please use the format: /donor [Donor] [Country], e.g. /donor EU Ethiopia"
            else:
                try:
                    reply = qa_chain.run(question)
                except Exception as e:
                    reply = f"❌ Error answering from donor document: {e}"

            requests.post(f"{TELEGRAM_API_URL}/sendMessage", json={
                "chat_id": chat_id,
                "text": reply
            })
            return {"ok": True}

        # Fallback reply
        requests.post(f"{TELEGRAM_API_URL}/sendMessage", json={
            "chat_id": chat_id,
            "text": "Hi! Try /donor EU Ethiopia to get a document-based strategy summary."
        })
        return {"ok": True}

    except Exception as e:
        print("Webhook error:", e)
        return {"ok": False, "error": str(e)}, 500
