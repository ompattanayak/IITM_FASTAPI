from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from chromadb import Client
from chromadb.config import Settings
import requests
import os
from dotenv import load_dotenv

load_dotenv()

AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
AIPIPE_URL = "https://aipipe.org/openai/v1/chat/completions"
EMBED_URL = "https://aipipe.org/openai/v1/embeddings"
MODEL = "openai/gpt-4.1-nano"
EMBED_MODEL = "text-embedding-ada-002"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

chroma_client = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chromadb"))
collection = chroma_client.get_or_create_collection("typescript_docs")

def embed_query(query: str):
    response = requests.post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"},
        json={"input": [query], "model": EMBED_MODEL}
    )
    return response.json()["data"][0]["embedding"]

@app.get("/search")
def search(q: str):
    embedding = embed_query(q)
    results = collection.query(query_embeddings=[embedding], n_results=5)

    context_chunks = results["documents"][0]
    sources = results["metadatas"][0]

    context_text = "\n---\n".join(context_chunks)

    prompt = f"""Use the following documentation excerpts to answer the question.

Documentation:
{context_text}

Question: {q}

Respond with a precise factual answer with supporting excerpt."""

    response = requests.post(
        AIPIPE_URL,
        headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"},
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a documentation assistant for TypeScript."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
    )

    if response.status_code == 200:
        answer = response.json()["choices"][0]["message"]["content"]
        return {
            "answer": answer,
            "sources": sources
        }
    else:
        return {"error": response.text}
