import os
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from git import Repo
import openai
import chromadb
from chromadb.config import Settings

# Config
REPO_URL = "https://github.com/basarat/typescript-book.git"
LOCAL_PATH = "typescript-book"
CHROMA_COLLECTION_NAME = "typescript-book"

# Set OpenAI API key and base URL for AIPipe
openai.api_key = os.getenv("AIPIPE_TOKEN")
openai.api_base = "https://aipipe.org/openrouter/v1"

# Clone repo if not exists
if not os.path.exists(LOCAL_PATH):
    print("Cloning repo...")
    Repo.clone_from(REPO_URL, LOCAL_PATH)
    print("Repo cloned.")

# Load and chunk markdown files
documents = []
md_files = list(Path(LOCAL_PATH).rglob("*.md"))
for md_file in md_files:
    text = md_file.read_text(encoding="utf-8", errors="ignore")
    # split on double newlines and keep chunks longer than 40 chars
    chunks = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
    for chunk in chunks:
        documents.append({"text": chunk, "source": str(md_file)})

print(f"Loaded {len(documents)} chunks.")

# Setup ChromaDB client and collection
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=".chroma"
))
collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

def embed_texts(texts):
    # Embed a list of texts using OpenAI embeddings via AIPipe
    response = openai.embeddings.create(
        model="openai/gpt-4.1-nano",
        input=texts
    )
    return [item.embedding for item in response.data]

# Populate collection if empty
if collection.count() == 0:
    print("Creating embeddings and populating ChromaDB...")
    texts = [doc["text"] for doc in documents]
    metadatas = [{"source": doc["source"]} for doc in documents]

    embeddings = embed_texts(texts)
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=[str(i) for i in range(len(texts))],
        embeddings=embeddings
    )
    print("ChromaDB populated.")
else:
    print("ChromaDB already has data.")

# FastAPI app setup
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/search")
def search(q: str = Query(..., description="Your search query")):
    query_embedding = embed_texts([q])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    matches = [
        {
            "text": doc,
            "source": meta["source"]
        }
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

    return {
        "answer": matches[0]["text"] if matches else "No relevant content found.",
        "sources": [m["source"] for m in matches]
    }
